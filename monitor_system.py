#!/usr/bin/env python3
"""
系统监控脚本
实时显示WebSocket和聚合器状态
"""

import subprocess
import time
import psutil
import os
from datetime import datetime
import glob
from pathlib import Path

def get_process_info(process_name):
    """获取进程信息"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process_name in ' '.join(proc.info['cmdline'] or []):
                return proc.info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def get_file_count(pattern):
    """获取文件数量"""
    files = glob.glob(pattern)
    return len(files)

def get_directory_size(path):
    """获取目录大小"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # MB
    except:
        return 0

def get_latest_file_info(pattern):
    """获取最新文件信息"""
    files = glob.glob(pattern)
    if not files:
        return None
    
    latest_file = max(files, key=os.path.getmtime)
    file_size = os.path.getsize(latest_file) / 1024  # KB
    mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
    
    return {
        'file': latest_file,
        'size_kb': file_size,
        'mod_time': mod_time
    }

def main():
    """主监控循环"""
    print("🚀 系统监控启动...")
    print("=" * 80)
    
    while True:
        try:
            # 清屏
            os.system('clear' if os.name == 'posix' else 'cls')
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"📊 系统状态监控 - {current_time}")
            print("=" * 80)
            
            # WebSocket进程状态
            print("🔌 WebSocket进程状态:")
            websocket_proc = get_process_info("okx_websocket")
            if websocket_proc:
                print(f"  ✅ 运行中 (PID: {websocket_proc['pid']})")
                try:
                    proc = psutil.Process(websocket_proc['pid'])
                    cpu_percent = proc.cpu_percent()
                    memory_mb = proc.memory_info().rss / (1024 * 1024)
                    print(f"  💻 CPU: {cpu_percent:.1f}% | 内存: {memory_mb:.1f}MB")
                except:
                    print("  ⚠️  无法获取进程详情")
            else:
                print("  ❌ 未运行")
            
            # 聚合器进程状态
            print("\n🔄 聚合器进程状态:")
            aggregator_proc = get_process_info("realtime_aggregator")
            if aggregator_proc:
                print(f"  ✅ 运行中 (PID: {aggregator_proc['pid']})")
                try:
                    proc = psutil.Process(aggregator_proc['pid'])
                    cpu_percent = proc.cpu_percent()
                    memory_mb = proc.memory_info().rss / (1024 * 1024)
                    print(f"  💻 CPU: {cpu_percent:.1f}% | 内存: {memory_mb:.1f}MB")
                except:
                    print("  ⚠️  无法获取进程详情")
            else:
                print("  ❌ 未运行")
            
            # 数据统计
            print("\n📁 数据统计:")
            
            # WebSocket原始数据
            websocket_files = get_file_count("data/websocket/orderbook_*.parquet")
            websocket_size = get_directory_size("data/websocket")
            print(f"  📊 WebSocket原始数据: {websocket_files} 文件, {websocket_size:.1f}MB")
            
            # 实时特征数据
            feature_files = get_file_count("data/realtime_features/*.parquet")
            feature_size = get_directory_size("data/realtime_features")
            print(f"  🎯 实时特征数据: {feature_files} 文件, {feature_size:.1f}MB")
            
            # 最新文件信息
            print("\n📄 最新文件信息:")
            
            # 最新OrderBook文件
            latest_ob = get_latest_file_info("data/websocket/orderbook_*.parquet")
            if latest_ob:
                print(f"  📊 最新OrderBook: {latest_ob['file'].split('/')[-1]}")
                print(f"     大小: {latest_ob['size_kb']:.1f}KB | 修改时间: {latest_ob['mod_time'].strftime('%H:%M:%S')}")
            
            # 最新特征文件
            latest_feature = get_latest_file_info("data/realtime_features/*.parquet")
            if latest_feature:
                print(f"  🎯 最新特征: {latest_feature['file'].split('/')[-1]}")
                print(f"     大小: {latest_feature['size_kb']:.1f}KB | 修改时间: {latest_feature['mod_time'].strftime('%H:%M:%S')}")
            
            # 系统资源
            print("\n💻 系统资源:")
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            print(f"  🖥️  CPU使用率: {cpu_percent:.1f}%")
            print(f"  🧠 内存使用率: {memory.percent:.1f}% ({memory.used//(1024**3):.1f}GB/{memory.total//(1024**3):.1f}GB)")
            print(f"  💾 磁盘使用率: {disk.percent:.1f}% ({disk.used//(1024**3):.1f}GB/{disk.total//(1024**3):.1f}GB)")
            
            # 日志文件大小
            print("\n📝 日志文件:")
            log_files = {
                'okx_websocket_optimized.log': 'WebSocket日志',
                'realtime_aggregator.log': '聚合器日志',
                'monitor_system.log': '监控日志'
            }
            
            for log_file, description in log_files.items():
                if os.path.exists(log_file):
                    size_kb = os.path.getsize(log_file) / 1024
                    print(f"  📄 {description}: {size_kb:.1f}KB")
                else:
                    print(f"  ❌ {description}: 不存在")
            
            print("\n" + "=" * 80)
            print("🔄 30秒后刷新... (Ctrl+C 退出)")
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 监控已停止")
            break
        except Exception as e:
            print(f"❌ 监控异常: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main() 