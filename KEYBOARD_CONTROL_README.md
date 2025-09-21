# ROSbot键盘控制系统使用说明

本系统提供了一个完整的键盘控制解决方案，可以控制运行在另一个Docker容器中的ROSbot机器人。基于Webots外部控制器协议，使用轮速百分比控制方式。

## 系统架构

```
┌─────────────────┐    TCP协议     ┌─────────────────┐    Webots协议   ┌─────────────────┐
│   控制端        │ ──────────────► │   容器适配器    │ ──────────────► │   ROSbot仿真    │
│ keyboard_control│  轮速百分比    │rosbot_container │  电机控制命令  │   (Webots)      │
│     (本机)      │   [0.0-1.0]    │   _adapter      │   [0-26rad/s]   │                 │
└─────────────────┘                └─────────────────┘                └─────────────────┘
```

## 文件说明

- `keyboard_control.py` - 键盘控制客户端程序（支持轮速百分比控制）
- `rosbot_container_adapter.py` - ROSbot容器适配器（在容器中运行）
- `robot_tcp_server.py` - 通用机器人TCP服务器程序（可选）
- `Dockerfile.robot` - 机器人容器Docker配置
- `docker-compose.robot.yml` - Docker Compose配置

## 快速开始

### 1. 启动机器人容器

#### 方法一：使用ROSbot容器适配器（推荐）

```bash
# 直接在容器中运行适配器
docker run -it --rm \
  --name rosbot-adapter \
  --network host \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  -w /workspace \
  python:3.9 \
  python rosbot_container_adapter.py --port 8888

# 或者使用自定义Webots URL
docker run -it --rm \
  --name rosbot-adapter \
  --network host \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  -w /workspace \
  python:3.9 \
  python rosbot_container_adapter.py --port 8888 --webots-url "tcp://localhost:1234"
```

#### 方法二：使用Docker Compose

```bash
# 构建并启动容器
docker-compose -f docker-compose.robot.yml up --build

# 后台运行
docker-compose -f docker-compose.robot.yml up -d --build

# 查看日志
docker-compose -f docker-compose.robot.yml logs -f rosbot-server
```

#### 方法三：本地运行适配器

```bash
# 直接在本地运行（需要Webots环境）
python rosbot_container_adapter.py --port 8888

# 指定Webots外部控制器URL
python rosbot_container_adapter.py --port 8888 --webots-url "tcp://localhost:1234"
```

### 2. 获取容器IP地址

容器启动后会自动显示IP地址，或者手动查询：

```bash
# 查看容器IP
docker inspect rosbot-control-server | grep IPAddress

# 或者进入容器查看
docker exec -it rosbot-control-server hostname -I
```

### 3. 启动键盘控制

在本机运行键盘控制程序：

```bash
# 连接到容器适配器
python keyboard_control.py --ip <容器IP> --port 8888

# 示例：连接到本地容器
python keyboard_control.py --ip 172.17.0.2 --port 8888

# 如果容器使用host网络模式
python keyboard_control.py --ip localhost --port 8888

# 直接连接Webots（不通过容器）
python keyboard_control.py --direct-webots --webots-url "tcp://localhost:1234"
```

## 控制说明

### 键盘控制键位

**移动控制（轮速百分比控制）：**
- `W/↑` - 前进（增加两轮速度）
- `S/↓` - 后退（减少两轮速度）
- `A/←` - 左转（左轮慢，右轮快）
- `D/→` - 右转（左轮快，右轮慢）
- `Q` - 左前（右轮增加更多）
- `E` - 右前（左轮增加更多）
- `Z` - 左后（左轮减少更多）
- `C` - 右后（右轮减少更多）

**轮速控制：**
- `+/=` - 增加轮速增量
- `-/_` - 减少轮速增量
- `空格` - 紧急停止

**其他功能：**
- `H` - 显示帮助信息
- `R` - 重置轮速
- `ESC` - 退出程序

**控制原理：**
- 轮速范围：0.0-1.0（对应0-26.0 rad/s）
- 差分驱动：通过左右轮速度差实现转向
- 平滑控制：限制单步速度变化，避免突变

### 命令行参数

#### keyboard_control.py 参数

```bash
python keyboard_control.py [选项]

选项:
  --ip IP地址              机器人容器IP地址 (默认: localhost)
  --port 端口              TCP通信端口 (默认: 8888)
  --webots-url URL         Webots外部控制器URL (可选)
  --direct-webots          直接连接Webots而非通过TCP服务器
  --max-motor-speed 速度   最大电机速度 (默认: 26.0 rad/s)
  --wheel-base 轮距        轮距 (默认: 0.22 m)
  --wheel-radius 半径      轮半径 (默认: 0.043 m)
  --speed-increment 增量   轮速增量 (默认: 0.1)
```

#### rosbot_container_adapter.py 参数

```bash
python rosbot_container_adapter.py [选项]

选项:
  --port 端口              监听端口 (默认: 8888)
  --webots-url URL         Webots外部控制器URL (可选)
  --robot-name 名称        机器人名称 (默认: rosbot)
  --max-motor-speed 速度   最大电机速度 (默认: 26.0 rad/s)
  --wheel-radius 半径      轮半径 (默认: 0.043 m)
  --wheel-base 轮距        轮距 (默认: 0.22 m)
```

## 部署方案

### 方案一：本地开发环境

```bash
# 1. 启动ROSbot容器适配器（在一个终端）
python rosbot_container_adapter.py --port 8888

# 2. 启动键盘控制（在另一个终端）
python keyboard_control.py --ip localhost --port 8888
```

### 方案二：Docker容器部署

```bash
# 1. 启动容器适配器
docker run -it --rm --name rosbot-adapter --network host -p 8888:8888 \
  -v $(pwd):/workspace -w /workspace python:3.9 \
  python rosbot_container_adapter.py --port 8888

# 2. 获取容器IP（如果不使用host网络）
ROBOT_IP=$(docker inspect rosbot-adapter | grep '"IPAddress"' | head -1 | awk -F'"' '{print $4}')

# 3. 启动控制
python keyboard_control.py --ip ${ROBOT_IP:-localhost} --port 8888
```

### 方案三：Webots外部控制器模式

```bash
# 1. 启动Webots仿真（在Webots中选择外部控制器模式）
# 2. 启动容器适配器连接到Webots
python rosbot_container_adapter.py --port 8888 --webots-url "tcp://localhost:1234"

# 3. 启动键盘控制
python keyboard_control.py --ip localhost --port 8888
```

### 方案四：跨主机部署

```bash
# 在机器人主机上启动容器适配器
python rosbot_container_adapter.py --port 8888

# 在控制主机上连接
python keyboard_control.py --ip <机器人主机IP> --port 8888
```

## 网络配置

### 端口说明

- `8888` - TCP控制通信端口
- `11311` - ROS Master端口（如果使用ROS）
- `1234` - Webots仿真端口（可选）

### 防火墙设置

如果跨主机部署，需要开放相应端口：

```bash
# Ubuntu/Debian
sudo ufw allow 8888
sudo ufw allow 11311

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8888/tcp
sudo firewall-cmd --permanent --add-port=11311/tcp
sudo firewall-cmd --reload
```

## 故障排除

### 常见问题

1. **连接失败**
   ```
   解决方案：
   - 检查容器是否正常运行
   - 确认IP地址和端口正确
   - 检查防火墙设置
   ```

2. **控制无响应**
   ```
   解决方案：
   - 检查命令超时设置
   - 确认机器人服务器日志
   - 重启服务器程序
   ```

3. **ROS连接问题**
   ```
   解决方案：
   - 检查ROS_MASTER_URI设置
   - 确认roscore是否运行
   - 使用TCP模式作为备选
   ```

### 调试命令

```bash
# 查看容器状态
docker ps
docker logs rosbot-control-server

# 测试网络连接
ping <容器IP>
telnet <容器IP> 8888

# 查看端口占用
netstat -tlnp | grep 8888
```

## 扩展功能

### 添加新的控制命令

在 `robot_tcp_server.py` 中的 `_process_command` 方法中添加新的命令处理逻辑：

```python
def _process_command(self, command_str: str):
    command = json.loads(command_str)
    
    # 添加新命令
    if command.get('type') == 'custom_action':
        self._handle_custom_action(command)
```

### 集成其他传感器

可以在服务器中添加传感器数据发布功能：

```python
# 添加传感器数据发布
self.sensor_pub = rospy.Publisher('/sensor_data', SensorData, queue_size=10)
```

## 安全注意事项

1. **网络安全**
   - 在生产环境中使用加密通信
   - 限制访问IP范围
   - 使用VPN连接

2. **机器人安全**
   - 设置合理的速度限制
   - 实现紧急停止功能
   - 添加碰撞检测

3. **系统安全**
   - 定期更新依赖包
   - 使用非root用户运行
   - 限制容器权限

## 许可证

本项目遵循MIT许可证。详见LICENSE文件。
