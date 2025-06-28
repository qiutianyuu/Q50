#!/usr/bin/env python3
"""
WebSocket数据收集监控脚本
实时显示数据收集进度和统计信息
"""

import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse

def get_websocket_stats(symbol: str = "ETH-USDT"):
    """获取WebSocket数据收集统计"""
    websocket_dir = "data/websocket"
    
    # 统计文件数量
    orderbook_files = glob.glob(f"{websocket_dir}/orderbook_{symbol}_*.parquet")
    trades_files = glob.glob(f"{websocket_dir}/trades_{symbol}_*.parquet")
    
    # 统计数据量
    total_orderbook_rows = 0
    total_trades_rows = 0
    
    for file in orderbook_files:
        try:
            df = pd.read_parquet(file)
            total_orderbook_rows += len(df)
        except:
            pass
    
    for file in trades_files:
        try:
            df = pd.read_parquet(file)
            total_trades_rows += len(df)
        except:
            pass
    
    # 获取时间范围
    all_files = orderbook_files + trades_files
    timestamps = []
    
    for file in all_files:
        try:
            # 从文件名提取时间戳
            filename = os.path.basename(file)
            if '_20250625_' in filename:
                time_str = filename.split('_20250625_')[1].split('.')[0]
                timestamps.append(datetime.strptime(f"20250625_{time_str}", "%Y%m%d_%H%M%S"))
        except:
            pass
    
    if timestamps:
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = end_time - start_time
    else:
        start_time = end_time = duration = None
    
    # 计算数据速率
    if duration and duration.total_seconds() > 0:
        orderbook_rate = total_orderbook_rows / duration.total_seconds() * 60  # 每分钟
        trades_rate = total_trades_rows / duration.total_seconds() * 60  # 每分钟
    else:
        orderbook_rate = trades_rate = 0
    
    return {
        'orderbook_files': len(orderbook_files),
        'trades_files': len(trades_files),
        'total_orderbook_rows': total_orderbook_rows,
        'total_trades_rows': total_trades_rows,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'orderbook_rate': orderbook_rate,
        'trades_rate': trades_rate
    }

def format_duration(duration):
    """格式化时长"""
    if duration is None:
        return "N/A"
    
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def main():
    parser = argparse.ArgumentParser(description="WebSocket数据收集监控")
    parser.add_argument("--symbol", default="ETH-USDT", help="交易对")
    parser.add_argument("--interval", type=int, default=30, help="刷新间隔(秒)")
    parser.add_argument("--once", action="store_true", help="只显示一次")
    
    args = parser.parse_args()
    
    print(f"🔍 WebSocket数据收集监控 - {args.symbol}")
    print("=" * 60)
    
    while True:
        try:
            stats = get_websocket_stats(args.symbol)
            
            # 清屏
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"🔍 WebSocket数据收集监控 - {args.symbol}")
            print("=" * 60)
            print(f"📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # 文件统计
            print("📁 文件统计:")
            print(f"  OrderBook文件: {stats['orderbook_files']} 个")
            print(f"  Trades文件: {stats['trades_files']} 个")
            print(f"  总文件数: {stats['orderbook_files'] + stats['trades_files']} 个")
            print()
            
            # 数据量统计
            print("📊 数据量统计:")
            print(f"  OrderBook数据: {stats['total_orderbook_rows']:,} 条")
            print(f"  Trades数据: {stats['total_trades_rows']:,} 条")
            print(f"  总数据量: {stats['total_orderbook_rows'] + stats['total_trades_rows']:,} 条")
            print()
            
            # 时间统计
            print("⏰ 时间统计:")
            if stats['start_time']:
                print(f"  开始时间: {stats['start_time'].strftime('%H:%M:%S')}")
                print(f"  结束时间: {stats['end_time'].strftime('%H:%M:%S')}")
                print(f"  运行时长: {format_duration(stats['duration'])}")
            else:
                print("  暂无数据")
            print()
            
            # 速率统计
            print("🚀 数据速率:")
            print(f"  OrderBook: {stats['orderbook_rate']:.1f} 条/分钟")
            print(f"  Trades: {stats['trades_rate']:.1f} 条/分钟")
            print()
            
            # 预估
            if stats['duration'] and stats['duration'].total_seconds() > 0:
                total_rate = stats['orderbook_rate'] + stats['trades_rate']
                if total_rate > 0:
                    # 预估达到目标数据量需要的时间
                    target_orderbook = 200000  # 20万条OrderBook
                    target_trades = 100000     # 10万条Trades
                    
                    remaining_orderbook = max(0, target_orderbook - stats['total_orderbook_rows'])
                    remaining_trades = max(0, target_trades - stats['total_trades_rows'])
                    
                    if stats['orderbook_rate'] > 0:
                        eta_orderbook = remaining_orderbook / stats['orderbook_rate']  # 分钟
                    else:
                        eta_orderbook = float('inf')
                    
                    if stats['trades_rate'] > 0:
                        eta_trades = remaining_trades / stats['trades_rate']  # 分钟
                    else:
                        eta_trades = float('inf')
                    
                    eta = max(eta_orderbook, eta_trades)
                    
                    print("🎯 目标进度:")
                    print(f"  OrderBook目标: 200,000 条 ({stats['total_orderbook_rows']/200000*100:.1f}%)")
                    print(f"  Trades目标: 100,000 条 ({stats['total_trades_rows']/100000*100:.1f}%)")
                    if eta < float('inf'):
                        print(f"  预估完成时间: {eta:.0f} 分钟")
                    else:
                        print("  预估完成时间: 未知")
                    print()
            
            # 状态指示
            print("📈 状态:")
            if stats['total_orderbook_rows'] > 0 and stats['total_trades_rows'] > 0:
                print("  ✅ 数据收集正常")
            elif stats['total_orderbook_rows'] > 0 or stats['total_trades_rows'] > 0:
                print("  ⚠️  部分数据收集中")
            else:
                print("  ❌ 无数据收集")
            
            if args.once:
                break
                
            print(f"\n🔄 {args.interval}秒后刷新... (Ctrl+C 退出)")
            time.sleep(args.interval)
            
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
            break
        except Exception as e:
            print(f"\n❌ 监控错误: {e}")
            if args.once:
                break
            time.sleep(args.interval)

if __name__ == "__main__":
    main() 