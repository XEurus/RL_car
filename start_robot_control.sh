#!/bin/bash

# ROSbot键盘控制系统启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
}

# 检查Python依赖
check_python_deps() {
    print_info "检查Python依赖..."
    
    python3 -c "import numpy, pygame" 2>/dev/null || {
        print_warning "缺少Python依赖，正在安装..."
        pip3 install numpy pygame
    }
}

# 启动机器人容器
start_robot_container() {
    print_info "启动机器人控制容器..."
    
    # 停止可能存在的旧容器
    docker-compose -f docker-compose.robot.yml down 2>/dev/null || true
    
    # 启动新容器
    docker-compose -f docker-compose.robot.yml up -d --build
    
    # 等待容器启动
    print_info "等待容器启动..."
    sleep 5
    
    # 检查容器状态
    if docker-compose -f docker-compose.robot.yml ps | grep -q "Up"; then
        print_success "机器人容器启动成功"
    else
        print_error "机器人容器启动失败"
        docker-compose -f docker-compose.robot.yml logs
        exit 1
    fi
}

# 获取容器IP
get_container_ip() {
    # 尝试获取容器IP
    CONTAINER_IP=$(docker inspect rosbot-control-server 2>/dev/null | grep '"IPAddress"' | head -1 | awk -F'"' '{print $4}' || echo "")
    
    if [ -z "$CONTAINER_IP" ] || [ "$CONTAINER_IP" = "null" ]; then
        # 如果使用host网络模式，使用localhost
        CONTAINER_IP="localhost"
    fi
    
    print_success "容器IP地址: $CONTAINER_IP"
    echo "$CONTAINER_IP"
}

# 启动键盘控制
start_keyboard_control() {
    local robot_ip=$1
    
    print_info "启动键盘控制程序..."
    print_info "连接地址: $robot_ip:8888"
    print_info "按Ctrl+C退出控制程序"
    
    # 启动键盘控制
    python3 keyboard_control.py --tcp --ip "$robot_ip" --port 8888
}

# 显示使用帮助
show_help() {
    echo "ROSbot键盘控制系统启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start           启动完整系统（默认）"
    echo "  container-only  仅启动机器人容器"
    echo "  control-only    仅启动键盘控制（需要指定IP）"
    echo "  stop            停止所有容器"
    echo "  status          查看系统状态"
    echo "  logs            查看容器日志"
    echo "  ip              获取容器IP地址"
    echo "  help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 启动完整系统"
    echo "  $0 container-only     # 仅启动容器"
    echo "  $0 control-only 192.168.1.100  # 连接到指定IP"
    echo "  $0 stop               # 停止系统"
}

# 停止系统
stop_system() {
    print_info "停止机器人控制系统..."
    docker-compose -f docker-compose.robot.yml down
    print_success "系统已停止"
}

# 查看系统状态
show_status() {
    print_info "系统状态:"
    docker-compose -f docker-compose.robot.yml ps
    
    print_info "容器日志（最后10行）:"
    docker-compose -f docker-compose.robot.yml logs --tail=10
}

# 查看日志
show_logs() {
    print_info "查看容器日志（按Ctrl+C退出）:"
    docker-compose -f docker-compose.robot.yml logs -f
}

# 主函数
main() {
    local action=${1:-start}
    
    case $action in
        "start")
            print_info "启动ROSbot键盘控制系统..."
            check_docker
            check_python_deps
            start_robot_container
            robot_ip=$(get_container_ip)
            
            print_success "系统启动完成！"
            echo ""
            echo "连接信息:"
            echo "  机器人IP: $robot_ip"
            echo "  控制端口: 8888"
            echo "  控制命令: python3 keyboard_control.py --tcp --ip $robot_ip --port 8888"
            echo ""
            
            read -p "是否立即启动键盘控制？(y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                start_keyboard_control "$robot_ip"
            else
                print_info "您可以稍后手动启动键盘控制："
                print_info "python3 keyboard_control.py --tcp --ip $robot_ip --port 8888"
            fi
            ;;
            
        "container-only")
            print_info "仅启动机器人容器..."
            check_docker
            start_robot_container
            robot_ip=$(get_container_ip)
            print_success "容器启动完成，IP: $robot_ip"
            ;;
            
        "control-only")
            local target_ip=${2:-localhost}
            print_info "启动键盘控制，连接到: $target_ip"
            check_python_deps
            start_keyboard_control "$target_ip"
            ;;
            
        "stop")
            stop_system
            ;;
            
        "status")
            show_status
            ;;
            
        "logs")
            show_logs
            ;;
            
        "ip")
            robot_ip=$(get_container_ip)
            echo "$robot_ip"
            ;;
            
        "help"|"-h"|"--help")
            show_help
            ;;
            
        *)
            print_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
