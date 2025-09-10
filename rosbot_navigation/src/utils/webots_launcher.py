# -*- coding: utf-8 -*-
"""
Webots 实例启动器（并行多实例）
- 为每个环境进程启动独立的 Webots 进程（FAST/无渲染/批处理）
- 将 world 文件中的控制器改为 extern，以便使用外部控制器（本进程的 Python controller）连接
- 解析 Webots 打印的 extern controller URL，并返回给上层环境构造函数

注意：该模块设计为在 SubprocVecEnv 的子进程中调用，避免 Popen 句柄在主进程中传递。
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

_URL_PATTERN = re.compile(r"url:\s*(?P<url>.+)$")


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    # 假定项目根目录包含 'warehouse/worlds/warehouse.wbt'
    for p in [here.parents[3], here.parents[2], here.parents[1]]:
        if (p / 'warehouse' / 'worlds' / 'warehouse.wbt').exists():
            return p
    # 回退到三层上级
    return here.parents[3]


def _prepare_extern_world(base_world: Path, instance_id: int) -> Path:
    """
    复制 world 到临时目录，将 controller 替换为 extern，并复制相关资源（如PROTO和纹理），
    以确保相对路径有效。返回临时 world 路径。
    """
    with open(base_world, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经是 extern controller
    if '<extern>' in content:
        print(f"📋 World 文件已包含 extern controller，无需替换")
    else:
        # 将 controller 改为 extern（支持多种格式）
        content = content.replace('controller "warehouse_navigation"', 'controller "<extern>"')
        content = content.replace("controller 'warehouse_navigation'", 'controller "<extern>"')
        print(f"📋 已将 controller 替换为 extern")
    
    # 添加缺失的 EXTERNPROTO 声明（解决 Astra 等缺失问题）
    if "Astra" in content and "EXTERNPROTO" not in content:
        proto_header = """
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/devices/orbbec/protos/Astra.proto"
"""
        content = proto_header + content
        print(f"📦 添加了 Astra PROTO 声明")
    
    # 1. 创建临时实例根目录
    tmp_instance_dir = Path(tempfile.gettempdir()) / f"webots_multi_{instance_id}"
    if tmp_instance_dir.exists():
        shutil.rmtree(tmp_instance_dir)
    tmp_instance_dir.mkdir(parents=True)
    
    # 2. 创建 worlds 子目录
    tmp_worlds_dir = tmp_instance_dir / 'worlds'
    tmp_worlds_dir.mkdir(parents=True)
    
    # 3. 写入修改后的 world 文件（放到 worlds 子目录）
    out_world = tmp_worlds_dir / base_world.name
    with open(out_world, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 4. 复制 protos 目录（如果存在）
    try:
        original_protos_dir = base_world.parent.parent / 'protos'
        if original_protos_dir.exists():
            shutil.copytree(original_protos_dir, tmp_instance_dir / 'protos')
            print(f"📦 复制 protos 目录到实例 {instance_id}")
    except Exception as e:
        print(f"⚠️ 复制 protos 目录失败: {e}")
    
    # 5. 复制 images 目录（如果存在）
    try:
        original_images_dir = base_world.parent.parent / 'images'
        if original_images_dir.exists():
            shutil.copytree(original_images_dir, tmp_instance_dir / 'images')
            print(f"🖼️ 复制 images 目录到实例 {instance_id}")
    except Exception as e:
        print(f"⚠️ 复制 images 目录失败: {e}")
    
    print(f"📁 为实例 {instance_id} 创建临时 world: {out_world}")
    return out_world


def start_webots_instance(instance_id: int,
                          world_path: Optional[str] = None,
                          headless: bool = True,
                          fast_mode: bool = True,
                          no_rendering: bool = False,
                          batch: bool = False,
                          minimize: bool = False,
                          stdout: bool = False,
                          stderr: bool = False,
                          timeout_seconds: int = 120) -> Tuple[subprocess.Popen, str]:
    """
    启动一个 Webots 实例并返回 (进程句柄, extern controller URL)。
    
    参数:
        instance_id: 实例ID，用于端口分配和临时目录命名
        world_path: World文件路径
        headless: 无头模式（自动启用no_rendering, batch, minimize等）
        fast_mode: 快速模式，关闭实时同步
        no_rendering: 无渲染模式
        batch: 批处理模式
        minimize: 窗口最小化
        stdout/stderr: 标准输出/错误重定向
        timeout_seconds: 连接超时时间
        
    返回:
        (进程句柄, extern controller URL)
    """
    project_root = _find_project_root()
    base_world = Path(world_path) if world_path else (project_root / 'warehouse' / 'worlds' / 'warehouse.wbt')
    if not base_world.exists():
        raise FileNotFoundError(f"未找到 world 文件: {base_world}")

    # 为 extern 控制器准备副本
    extern_world = _prepare_extern_world(base_world, instance_id)

    # 构建命令
    cmd = []
    # 优先使用 xvfb-run（在无显示环境）
    if headless:
        cmd += ['xvfb-run', '--auto-servernum']
    cmd += ['webots']
    
    # 分别应用各参数，确保每个命令行选项都正确添加
    if fast_mode:
        cmd += ['--mode=fast']
    
    # 所有渲染相关参数单独处理，确保能正确应用
    if headless or no_rendering:
        cmd += ['--no-rendering']
    if headless or batch:
        cmd += ['--batch']
    if headless or minimize:
        cmd += ['--minimize']
    
    # 为每个实例分配不同的端口（避免冲突）- 使用更大的间隔以避免端口扫描冲突
    port = 10000 + (instance_id * 100)  # 每个实例间隔100，给足缓冲
    cmd += [f'--port={port}']
    
    # 输出重定向选项
    if stdout or headless:
        cmd += ['--stdout']
    if stderr or headless:
        cmd += ['--stderr']
    
    # 添加world路径
    cmd += [str(extern_world)]

    # 绑定到单核可选：由外部调用者使用 taskset 控制
    env = os.environ.copy()
    
    # 添加环境变量禁用GPU渲染，避免OpenGL问题
    env['LIBGL_ALWAYS_SOFTWARE'] = '1'  # 强制使用软件渲染
    env['WEBOTS_DISABLE_GPU'] = '1'

    # 打印详细的启动信息和启用的参数
    print(f"🚀 启动 Webots 实例 {instance_id}:")
    print(f"   - World: {extern_world}")
    print(f"   - 端口: {port}")
    print(f"   - 无头模式: {headless}")
    print(f"   - 快速模式: {fast_mode}")
    print(f"   - 无渲染: {headless or no_rendering}")
    print(f"   - 批处理: {headless or batch}")
    print(f"   - 最小化: {headless or minimize}")
    print(f"   - 完整命令: {' '.join(cmd)}")
    
    # 启动进程
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env
    )

    # 解析 extern URL
    url: Optional[str] = None
    start_time = time.time()
    assert proc.stdout is not None
    
    print(f"⏳ 等待 Webots 实例 {instance_id} 输出 extern controller URL...")
    
    all_output = []  # 收集所有输出用于调试
    while True:
        if (time.time() - start_time) > timeout_seconds:
            print(f"⚠️ 实例 {instance_id} 超时！使用默认URL")
            # 使用默认URL（端口基于instance_id）
            port = 10000 + (instance_id * 100)
            url = f"tcp://localhost:{port}?name=rosbot"
            print(f"✅ 实例 {instance_id} 使用默认URL: {url}")
            break
            
        line = proc.stdout.readline()
        if not line:
            # 进程可能已结束
            if proc.poll() is not None:
                print(f"❌ Webots 实例 {instance_id} 进程提前退出，返回码 {proc.returncode}")
                print("最后的输出：")
                for output_line in all_output[-5:]:
                    print(f"   📄 {output_line.rstrip()}")
                raise RuntimeError(f"Webots 进程提前退出，返回码 {proc.returncode}")
            time.sleep(0.05)
            continue
            
        all_output.append(line)
        print(f"🔍 实例 {instance_id} 输出: {line.rstrip()}")
        
        # 查找 URL - 改进正则表达式
        m = _URL_PATTERN.search(line)
        if m:
            url = m.group('url').strip()
            print(f"✅ 实例 {instance_id} 找到URL (正则): {url}")
            break
            
        # 兼容其他格式：包含 'extern controller' 且有 'ipc://' 或 'tcp://'
        if 'extern controller' in line and ('ipc://' in line or 'tcp://' in line):
            # 简易提取
            if 'url:' in line:
                parts = line.strip().split('url:')
                if len(parts) >= 2:
                    url = parts[-1].strip()
                    print(f"✅ 实例 {instance_id} 找到URL (分割): {url}")
                    break
            else:
                # 直接查找 ipc:// 或 tcp://
                if 'ipc://' in line:
                    start_idx = line.find('ipc://')
                    url_part = line[start_idx:].strip()
                    url = url_part.split()[0]  # 取第一个词
                    print(f"✅ 实例 {instance_id} 找到URL (直接): {url}")
                    break
                elif 'tcp://' in line:
                    start_idx = line.find('tcp://')
                    url_part = line[start_idx:].strip()
                    url = url_part.split()[0]  # 取第一个词
                    print(f"✅ 实例 {instance_id} 找到URL (直接): {url}")
                    break
        
        # 检查标准的 extern controller 等待消息 - 支持更多模式
        if ('extern controller' in line or 'extern controller:' in line) and ('Waiting for' in line or 'waiting for' in line):
            # 首先尝试匹配端口号
            import re
            port_match = re.search(r'port (\d+)', line)
            if port_match:
                port = port_match.group(1)
                # 构造连接URL (使用本地TCP连接)
                url = f"tcp://localhost:{port}"
                print(f"✅ 实例 {instance_id} 找到端口 (标准格式): {url}")
                break
            
            # 通用解析：尝试提取名称为'rosbot'的机器人标识
            if 'rosbot' in line:
                robot_match = re.search(r'robot named [\'"]([^\'"]+)[\'"]', line)
                if robot_match:
                    robot_name = robot_match.group(1)
                    # 从启动参数中获取端口
                    robot_port = 10000 + (instance_id * 100)
                    url = f"tcp://localhost:{robot_port}?name={robot_name}"
                    print(f"✅ 实例 {instance_id} 使用机器人名称构建URL: {url}")
                    break

    if not url:
        # 后备方案：生成一个可能有效的URL
        port = 10000 + (instance_id * 100)
        url = f"tcp://localhost:{port}?name=rosbot"
        print(f"⚠️ 实例 {instance_id} 未能解析到 extern controller URL，使用后备URL: {url}")
        print("最近10行输出：")
        for output_line in all_output[-10:]:
            print(f"   📄 {output_line.rstrip()}")
        # 继续使用此URL尝试连接，而不是失败

    print(f"🎯 实例 {instance_id} extern controller URL: {url}")
    return proc, url


def stop_webots_instance(proc: subprocess.Popen, wait_seconds: int = 5) -> None:
    """停止 Webots 实例。"""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


def attach_process_cleanup_to_env(env, proc: subprocess.Popen):
    """为环境对象挂接关闭回调，确保关闭环境时杀掉 Webots 进程。"""
    import types

    original_close = getattr(env, 'close', None)

    def _wrapped_close(self):
        try:
            if callable(original_close):
                original_close()
        finally:
            stop_webots_instance(proc)

    env.close = types.MethodType(_wrapped_close, env)
    # 以防万一，注册 atexit
    try:
        import atexit
        atexit.register(lambda: stop_webots_instance(proc))
    except Exception:
        pass
