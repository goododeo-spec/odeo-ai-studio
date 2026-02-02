#!/usr/bin/env python
"""
API 功能演示脚本
"""
import time
import requests
from datetime import datetime

API_BASE = "http://localhost:8080/api/v1"

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_section(text):
    """打印章节"""
    print(f"\n{text}")
    print("-" * 60)

def demo_gpu_monitoring():
    """演示 GPU 监控功能"""
    print_header("GPU 监控功能演示")

    try:
        # 1. 获取所有 GPU 状态
        print_section("1. 获取所有 GPU 状态")
        response = requests.get(f"{API_BASE}/gpu/status")
        data = response.json()

        print(f"状态码: {response.status_code}")
        print(f"消息: {data['message']}")

        gpus = data['data']['gpus']
        print(f"\n发现 {len(gpus)} 个 GPU")

        for gpu in gpus:
            print(f"\n  GPU {gpu['gpu_id']}: {gpu['name']}")
            print(f"    状态: {gpu['status']}")
            print(f"    显存: {gpu['memory']['used']}/{gpu['memory']['total']} MB "
                  f"({gpu['memory']['utilization']}%)")
            print(f"    利用率: {gpu['utilization_gpu']}%")
            print(f"    温度: {gpu['temperature']['gpu']}°C")
            if gpu['power_usage']:
                print(f"    功耗: {gpu['power_usage']}/{gpu['power_limit']} W")

        # 2. 获取可用 GPU
        print_section("2. 获取可用 GPU 列表")
        response = requests.get(f"{API_BASE}/gpu/available")
        data = response.json()

        available_gpus = data['data']['available_gpus']
        print(f"\n找到 {len(available_gpus)} 个可用 GPU")

        for gpu in available_gpus:
            print(f"\n  GPU {gpu['gpu_id']}: {gpu['name']}")
            print(f"    可用显存: {gpu['memory_free']} MB ({gpu['memory_free_gb']:.2f} GB)")

        # 3. 获取 GPU 汇总
        print_section("3. 获取 GPU 汇总信息")
        response = requests.get(f"{API_BASE}/gpu/summary")
        data = response.json()

        summary = data['data']
        print(f"\nGPU 汇总:")
        print(f"  总数: {summary['total_gpus']}")
        print(f"  可用: {summary['available_gpus']}")
        print(f"  忙碌: {summary['busy_gpus']}")
        print(f"  总显存: {summary['total_memory']} MB")
        print(f"  可用显存: {summary['available_memory']} MB")

        # 4. 实时监控 (5 秒)
        print_section("4. 实时监控 (5 秒)")
        for i in range(5):
            response = requests.get(f"{API_BASE}/gpu/status")
            data = response.json()

            gpus = data['data']['gpus']
            print(f"\n  监控点 {i+1} - {datetime.now().strftime('%H:%M:%S')}")

            for gpu in gpus:
                print(f"    GPU {gpu['gpu_id']}: 利用率 {gpu['utilization_gpu']}%, "
                      f"显存 {gpu['memory']['utilization']}%, "
                      f"温度 {gpu['temperature']['gpu']}°C")

            if i < 4:  # 最后一次不等待
                time.sleep(1)

        print("\n" + "=" * 60)
        print("  演示完成")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ 错误: 无法连接到 API 服务器")
        print("\n请先启动 API 服务器:")
        print("  cd /root/diffusion-pipe/api")
        print("  make dev")
        print("\n或使用后台模式:")
        print("  make background")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")

def demo_gpu_filtering():
    """演示 GPU 筛选功能"""
    print_header("GPU 筛选功能演示")

    try:
        # 按显存筛选
        print_section("1. 筛选可用显存 > 10GB 的 GPU")
        response = requests.get(f"{API_BASE}/gpu/available?min_memory=10000")
        data = response.json()

        available_gpus = data['data']['available_gpus']
        print(f"\n找到 {len(available_gpus)} 个符合要求的 GPU")

        for gpu in available_gpus:
            print(f"\n  GPU {gpu['gpu_id']}: {gpu['name']}")
            print(f"    可用显存: {gpu['memory_free']} MB")

        # 获取特定 GPU 详情
        print_section("2. 获取 GPU 0 详细信息")
        if available_gpus:
            gpu_id = available_gpus[0]['gpu_id']
        else:
            gpu_id = 0  # 默认 GPU 0

        response = requests.get(f"{API_BASE}/gpu/{gpu_id}/details")
        data = response.json()

        if response.status_code == 200:
            gpu = data['data']
            print(f"\nGPU {gpu['gpu_id']}: {gpu['name']}")
            print(f"  状态: {gpu['status']}")
            print(f"  可用: {'是' if gpu['is_available'] else '否'}")

            print(f"\n  显存信息:")
            print(f"    总计: {gpu['memory']['total']} MB")
            print(f"    已用: {gpu['memory']['used']} MB")
            print(f"    可用: {gpu['memory']['free']} MB")

            print(f"\n  性能信息:")
            print(f"    GPU 利用率: {gpu['utilization']['gpu']}%")
            print(f"    显存利用率: {gpu['utilization']['memory']}%")

            print(f"\n  温度:")
            print(f"    GPU: {gpu['temperature']['gpu']}°C")

            if gpu['current_task']:
                print(f"\n  当前任务:")
                print(f"    ID: {gpu['current_task']['task_id']}")
                print(f"    类型: {gpu['current_task']['task_type']}")
                print(f"    进度: {gpu['current_task']['progress']:.1f}%")

    except requests.exceptions.ConnectionError:
        print("\n❌ 错误: 无法连接到 API 服务器")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  Diffusion-Pipe API 功能演示")
    print("=" * 60)
    print("\n本演示将展示:")
    print("  1. GPU 状态监控")
    print("  2. 可用 GPU 查询")
    print("  3. GPU 筛选功能")
    print("  4. 实时监控")

    print("\n" + "=" * 60)
    input("按 Enter 键继续...")
    print("=" * 60)

    # 执行演示
    demo_gpu_monitoring()
    demo_gpu_filtering()

    print("\n" + "=" * 60)
    print("  演示结束")
    print("=" * 60)
    print("\n更多功能:")
    print("  - 查看 API 文档: /root/diffusion-pipe/TRAINING_API.md")
    print("  - 运行完整测试: make test")
    print("  - 查看源代码: /root/diffusion-pipe/api/")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
