#!/usr/bin/env python3
"""
选择最佳参数脚本
从回测结果中选择最佳参数组合
"""

import json
import sys
import yaml
import argparse

def select_best_params(results_file, metric='sharpe'):
    """从回测结果中选择最佳参数"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("No results found")
        return None
    
    # 按指定指标排序
    if metric == 'sharpe':
        best = max(results, key=lambda x: x.get('sharpe', -999))
    elif metric == 'total_return':
        best = max(results, key=lambda x: x.get('total_return', -999))
    elif metric == 'win_rate':
        best = max(results, key=lambda x: x.get('win_rate', -999))
    else:
        print(f"Unknown metric: {metric}")
        return None
    
    # 提取关键参数
    best_params = {
        'threshold_long': best.get('threshold_long', 0.7),
        'threshold_short': best.get('threshold_short', 0.3),
        'holding_steps': best.get('holding_steps', 60),
        'sharpe': best.get('sharpe', 0),
        'total_return': best.get('total_return', 0),
        'win_rate': best.get('win_rate', 0),
        'total_trades': best.get('total_trades', 0),
        'max_drawdown': best.get('max_drawdown', 0)
    }
    
    return best_params

def main():
    parser = argparse.ArgumentParser(description='Select Best Parameters')
    parser.add_argument('results_file', type=str, help='Backtest results JSON file')
    parser.add_argument('--metric', type=str, default='sharpe', 
                       choices=['sharpe', 'total_return', 'win_rate'], 
                       help='Metric to optimize for')
    parser.add_argument('--output', type=str, help='Output YAML file path')
    
    args = parser.parse_args()
    
    # 选择最佳参数
    best_params = select_best_params(args.results_file, args.metric)
    
    if best_params is None:
        sys.exit(1)
    
    # 输出到文件或标准输出
    yaml_str = yaml.safe_dump(best_params, default_flow_style=False, allow_unicode=True)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(yaml_str)
        print(f"Best parameters saved to: {args.output}")
    else:
        print(yaml_str)
    
    return best_params

if __name__ == "__main__":
    main() 