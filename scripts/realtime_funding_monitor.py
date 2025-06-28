#!/usr/bin/env python3
"""
实时资金费率监控器
监控当前funding rate，计算套利机会，发送交易信号
"""
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
from pathlib import Path

# 配置
THRESHOLD = 0.001  # 0.1% 阈值
HOLDING_HOURS = 8  # 持仓时间
MAKER_FEE = 0.0001  # 0.01% maker fee
TAKER_FEE = 0.0005  # 0.05% taker fee
SLIPPAGE = 0.0002   # 0.02% slippage
HEDGE_SPREAD = 0.0003  # 0.03% hedge spread

class FundingMonitor:
    def __init__(self):
        self.symbol = "ETH-USDT"
        self.exchanges = {
            "binance": "https://fapi.binance.com/fapi/v1/premiumIndex",
            "okx": "https://www.okx.com/api/v5/public/funding-rate",
            "bybit": "https://api.bybit.com/v5/market/funding/history"
        }
        self.history = []
        self.signals = []
        
    def fetch_binance_funding(self):
        """获取Binance资金费率"""
        try:
            url = f"{self.exchanges['binance']}?symbol={self.symbol.replace('-', '')}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'lastFundingRate' in data:
                return {
                    'exchange': 'binance',
                    'funding_rate': float(data['lastFundingRate']),
                    'timestamp': datetime.now(),
                    'next_funding_time': int(data['nextFundingTime'])
                }
        except Exception as e:
            print(f"Binance API错误: {e}")
        return None
    
    def fetch_okx_funding(self):
        """获取OKX资金费率"""
        try:
            url = f"{self.exchanges['okx']}?instId={self.symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['code'] == '0' and data['data']:
                funding_data = data['data'][0]
                return {
                    'exchange': 'okx',
                    'funding_rate': float(funding_data['fundingRate']),
                    'timestamp': datetime.now(),
                    'next_funding_time': int(funding_data['nextFundingTime'])
                }
        except Exception as e:
            print(f"OKX API错误: {e}")
        return None
    
    def fetch_bybit_funding(self):
        """获取Bybit资金费率"""
        try:
            url = f"{self.exchanges['bybit']}?category=linear&symbol={self.symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['retCode'] == 0 and data['result']['list']:
                funding_data = data['result']['list'][0]
                return {
                    'exchange': 'bybit',
                    'funding_rate': float(funding_data['fundingRate']),
                    'timestamp': datetime.now(),
                    'next_funding_time': int(funding_data['nextFundingTime'])
                }
        except Exception as e:
            print(f"Bybit API错误: {e}")
        return None
    
    def fetch_all_funding_rates(self):
        """获取所有交易所的资金费率"""
        results = []
        
        # 并行获取
        binance_data = self.fetch_binance_funding()
        okx_data = self.fetch_okx_funding()
        bybit_data = self.fetch_bybit_funding()
        
        for data in [binance_data, okx_data, bybit_data]:
            if data:
                results.append(data)
        
        return results
    
    def calculate_arbitrage_opportunity(self, funding_rates):
        """计算套利机会"""
        if not funding_rates:
            return None
        
        # 计算平均funding rate
        avg_funding = np.mean([fr['funding_rate'] for fr in funding_rates])
        
        # 计算套利机会
        opportunity = None
        
        if avg_funding > THRESHOLD:
            # 做多永续机会
            expected_return = avg_funding * HOLDING_HOURS / 8
            total_cost = (MAKER_FEE + SLIPPAGE) * 2 + HEDGE_SPREAD * 2
            net_return = expected_return - total_cost
            
            opportunity = {
                'type': 'long',
                'funding_rate': avg_funding,
                'expected_return': expected_return,
                'total_cost': total_cost,
                'net_return': net_return,
                'profitable': net_return > 0,
                'timestamp': datetime.now()
            }
            
        elif avg_funding < -THRESHOLD:
            # 做空永续机会
            expected_return = -avg_funding * HOLDING_HOURS / 8
            total_cost = (MAKER_FEE + SLIPPAGE) * 2 + HEDGE_SPREAD * 2
            net_return = expected_return - total_cost
            
            opportunity = {
                'type': 'short',
                'funding_rate': avg_funding,
                'expected_return': expected_return,
                'total_cost': total_cost,
                'net_return': net_return,
                'profitable': net_return > 0,
                'timestamp': datetime.now()
            }
        
        return opportunity
    
    def generate_signal(self, opportunity):
        """生成交易信号"""
        if not opportunity or not opportunity['profitable']:
            return None
        
        signal = {
            'timestamp': opportunity['timestamp'],
            'action': f"{opportunity['type'].upper()}_PERP",
            'funding_rate': opportunity['funding_rate'],
            'expected_return': opportunity['net_return'],
            'confidence': 'HIGH' if abs(opportunity['funding_rate']) > THRESHOLD * 2 else 'MEDIUM',
            'message': f"资金费率套利机会: {opportunity['type'].upper()} 永续合约, "
                      f"Funding: {opportunity['funding_rate']*100:.3f}%, "
                      f"预期收益: {opportunity['net_return']*100:.3f}%"
        }
        
        return signal
    
    def monitor_once(self):
        """执行一次监控"""
        print(f"\n🔄 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 监控中...")
        
        # 获取资金费率
        funding_rates = self.fetch_all_funding_rates()
        
        if not funding_rates:
            print("❌ 无法获取资金费率数据")
            return
        
        # 显示当前状态
        print("📊 当前资金费率:")
        for fr in funding_rates:
            print(f"  {fr['exchange'].upper()}: {fr['funding_rate']*100:.4f}%")
        
        # 计算套利机会
        opportunity = self.calculate_arbitrage_opportunity(funding_rates)
        
        if opportunity:
            print(f"\n🎯 发现套利机会!")
            print(f"类型: {opportunity['type'].upper()}")
            print(f"Funding Rate: {opportunity['funding_rate']*100:.4f}%")
            print(f"预期收益: {opportunity['expected_return']*100:.4f}%")
            print(f"交易成本: {opportunity['total_cost']*100:.4f}%")
            print(f"净收益: {opportunity['net_return']*100:.4f}%")
            print(f"是否盈利: {'✅' if opportunity['profitable'] else '❌'}")
            
            # 生成信号
            signal = self.generate_signal(opportunity)
            if signal:
                self.signals.append(signal)
                print(f"\n🚨 交易信号: {signal['message']}")
        else:
            print("📉 当前无套利机会")
        
        # 保存历史
        self.history.append({
            'timestamp': datetime.now(),
            'funding_rates': funding_rates,
            'opportunity': opportunity
        })
    
    def run_monitor(self, interval_minutes=5, max_iterations=None):
        """运行监控器"""
        print(f"🚀 启动资金费率监控器")
        print(f"监控间隔: {interval_minutes} 分钟")
        print(f"阈值: {THRESHOLD*100:.1f}%")
        print(f"持仓时间: {HOLDING_HOURS} 小时")
        
        iteration = 0
        
        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    break
                
                self.monitor_once()
                iteration += 1
                
                if max_iterations and iteration < max_iterations:
                    print(f"⏰ {interval_minutes} 分钟后再次检查...")
                    time.sleep(interval_minutes * 60)
                elif not max_iterations:
                    print(f"⏰ {interval_minutes} 分钟后再次检查...")
                    time.sleep(interval_minutes * 60)
                    
        except KeyboardInterrupt:
            print("\n⏹️ 监控器已停止")
            self.save_results()
    
    def save_results(self):
        """保存监控结果"""
        output_dir = Path("analysis/funding_arbitrage")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存历史数据
        if self.history:
            history_df = pd.DataFrame([
                {
                    'timestamp': h['timestamp'],
                    'avg_funding': np.mean([fr['funding_rate'] for fr in h['funding_rates']]) if h['funding_rates'] else None,
                    'opportunity_type': h['opportunity']['type'] if h['opportunity'] else None,
                    'opportunity_profitable': h['opportunity']['profitable'] if h['opportunity'] else None
                }
                for h in self.history
            ])
            history_df.to_csv(output_dir / 'realtime_monitor_history.csv', index=False)
            print(f"💾 历史数据已保存: {output_dir / 'realtime_monitor_history.csv'}")
        
        # 保存信号
        if self.signals:
            signals_df = pd.DataFrame(self.signals)
            signals_df.to_csv(output_dir / 'realtime_monitor_signals.csv', index=False)
            print(f"💾 交易信号已保存: {output_dir / 'realtime_monitor_signals.csv'}")
            
            print(f"\n📊 监控统计:")
            print(f"总监控次数: {len(self.history)}")
            print(f"生成信号数: {len(self.signals)}")
            if self.signals:
                long_signals = len([s for s in self.signals if 'LONG' in s['action']])
                short_signals = len([s for s in self.signals if 'SHORT' in s['action']])
                print(f"做多信号: {long_signals}")
                print(f"做空信号: {short_signals}")

def main():
    monitor = FundingMonitor()
    
    # 运行一次测试
    print("🧪 运行一次测试...")
    monitor.monitor_once()
    
    # 询问是否继续监控
    response = input("\n是否开始持续监控? (y/n): ")
    if response.lower() == 'y':
        interval = int(input("监控间隔(分钟): ") or "5")
        monitor.run_monitor(interval_minutes=interval)
    else:
        monitor.save_results()

if __name__ == "__main__":
    main() 