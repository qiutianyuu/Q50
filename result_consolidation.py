#!/usr/bin/env python3
"""
结果整合分析 - 48小时冲刺结果总结和决策建议
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_walk_forward_results():
    """分析Walk-Forward结果"""
    print("📊 分析Walk-Forward结果...")
    
    results = {}
    
    # 读取15m结果
    if Path('walk_forward_results_15m_selected.csv').exists():
        df_15m = pd.read_csv('walk_forward_results_15m_selected.csv')
        results['15m'] = {
            'mean_test_auc': df_15m['test_auc'].mean(),
            'std_test_auc': df_15m['test_auc'].std(),
            'mean_overfitting': df_15m['overfitting'].mean(),
            'stable_windows': len(df_15m[df_15m['test_auc'] > 0.55]),
            'total_windows': len(df_15m)
        }
    
    # 读取5m结果
    if Path('walk_forward_results_5m_selected.csv').exists():
        df_5m = pd.read_csv('walk_forward_results_5m_selected.csv')
        results['5m'] = {
            'mean_test_auc': df_5m['test_auc'].mean(),
            'std_test_auc': df_5m['test_auc'].std(),
            'mean_overfitting': df_5m['overfitting'].mean(),
            'stable_windows': len(df_5m[df_5m['test_auc'] > 0.55]),
            'total_windows': len(df_5m)
        }
    
    return results

def analyze_optuna_results():
    """分析Optuna优化结果"""
    print("🔍 分析Optuna优化结果...")
    
    results = {}
    
    # 读取15m优化结果
    if Path('optuna_results_15m.json').exists():
        with open('optuna_results_15m.json', 'r') as f:
            optuna_15m = json.load(f)
        results['15m'] = {
            'best_auc': optuna_15m['best_auc'],
            'n_trials': optuna_15m['n_trials'],
            'best_params': optuna_15m['best_params']
        }
    
    # 读取5m优化结果
    if Path('optuna_results_5m.json').exists():
        with open('optuna_results_5m.json', 'r') as f:
            optuna_5m = json.load(f)
        results['5m'] = {
            'best_auc': optuna_5m['best_auc'],
            'n_trials': optuna_5m['n_trials'],
            'best_params': optuna_5m['best_params']
        }
    
    return results

def analyze_backtest_results():
    """分析回测结果"""
    print("💰 分析回测结果...")
    
    results = {}
    
    # 读取15m回测结果
    if Path('trades_15m_selected.csv').exists():
        trades_15m = pd.read_csv('trades_15m_selected.csv')
        results['15m'] = {
            'total_trades': len(trades_15m),
            'win_rate': len(trades_15m[trades_15m['returns'] > 0]) / len(trades_15m),
            'total_returns': trades_15m['returns'].sum(),
            'avg_returns': trades_15m['returns'].mean(),
            'max_drawdown': trades_15m['returns'].min(),
            'avg_holding_hours': trades_15m['holding_hours'].mean()
        }
    
    # 读取5m回测结果
    if Path('trades_5m_selected.csv').exists():
        trades_5m = pd.read_csv('trades_5m_selected.csv')
        results['5m'] = {
            'total_trades': len(trades_5m),
            'win_rate': len(trades_5m[trades_5m['returns'] > 0]) / len(trades_5m),
            'total_returns': trades_5m['returns'].sum(),
            'avg_returns': trades_5m['returns'].mean(),
            'max_drawdown': trades_5m['returns'].min(),
            'avg_holding_hours': trades_5m['holding_hours'].mean()
        }
    
    return results

def analyze_feature_selection():
    """分析特征筛选结果"""
    print("🎯 分析特征筛选结果...")
    
    results = {}
    
    # 读取15m特征列表
    if Path('data/features_15m_selected_features.json').exists():
        with open('data/features_15m_selected_features.json', 'r') as f:
            features_15m = json.load(f)
        results['15m'] = {
            'total_features': features_15m['total_features'],
            'reduction': features_15m['reduction'],
            'selected_features': features_15m['selected_features']
        }
    
    # 读取5m特征列表
    if Path('data/features_5m_selected_features.json').exists():
        with open('data/features_5m_selected_features.json', 'r') as f:
            features_5m = json.load(f)
        results['5m'] = {
            'total_features': features_5m['total_features'],
            'reduction': features_5m['reduction'],
            'selected_features': features_5m['selected_features']
        }
    
    return results

def generate_decision_recommendations(walk_forward_results, optuna_results, backtest_results, feature_results):
    """生成决策建议"""
    print("🎯 生成决策建议...")
    
    recommendations = []
    
    # 分析模型性能
    if '15m' in walk_forward_results and '5m' in walk_forward_results:
        wf_15m = walk_forward_results['15m']
        wf_5m = walk_forward_results['5m']
        
        if wf_15m['mean_test_auc'] > wf_5m['mean_test_auc']:
            recommendations.append("✅ 15m模型表现更稳定，建议优先使用15m模型")
        else:
            recommendations.append("✅ 5m模型表现更稳定，建议优先使用5m模型")
        
        if wf_15m['mean_overfitting'] < 0.1 and wf_5m['mean_overfitting'] < 0.1:
            recommendations.append("✅ 两个模型过拟合程度都在可接受范围内")
        else:
            recommendations.append("⚠️ 模型存在过拟合风险，需要进一步正则化")
    
    # 分析回测结果
    if '15m' in backtest_results and '5m' in backtest_results:
        bt_15m = backtest_results['15m']
        bt_5m = backtest_results['5m']
        
        if bt_15m['total_returns'] > 0 or bt_5m['total_returns'] > 0:
            recommendations.append("✅ 模型在回测中显示盈利潜力")
        else:
            recommendations.append("⚠️ 回测结果为负收益，需要调整策略参数")
        
        if bt_15m['win_rate'] > 0.5 or bt_5m['win_rate'] > 0.5:
            recommendations.append("✅ 胜率超过50%，策略方向正确")
        else:
            recommendations.append("⚠️ 胜率偏低，需要优化信号生成逻辑")
    
    # 分析特征筛选
    if '15m' in feature_results and '5m' in feature_results:
        feat_15m = feature_results['15m']
        feat_5m = feature_results['5m']
        
        reduction_15m = feat_15m['reduction'] / (feat_15m['total_features'] + feat_15m['reduction'])
        reduction_5m = feat_5m['reduction'] / (feat_5m['total_features'] + feat_5m['reduction'])
        
        if reduction_15m > 0.7 and reduction_5m > 0.7:
            recommendations.append("✅ 特征筛选效果显著，大幅降低维度")
        else:
            recommendations.append("⚠️ 特征筛选效果有限，可能需要更严格的标准")
    
    return recommendations

def main():
    print("🚀 48小时冲刺结果整合分析")
    print("=" * 50)
    
    # 分析各项结果
    walk_forward_results = analyze_walk_forward_results()
    optuna_results = analyze_optuna_results()
    backtest_results = analyze_backtest_results()
    feature_results = analyze_feature_selection()
    
    # 生成决策建议
    recommendations = generate_decision_recommendations(
        walk_forward_results, optuna_results, backtest_results, feature_results
    )
    
    # 输出结果总结
    print("\n📊 结果总结")
    print("=" * 50)
    
    print("\n🔍 Walk-Forward验证结果:")
    for timeframe, results in walk_forward_results.items():
        print(f"  {timeframe}: AUC={results['mean_test_auc']:.4f}±{results['std_test_auc']:.4f}, "
              f"过拟合={results['mean_overfitting']:.4f}, 稳定窗口={results['stable_windows']}/{results['total_windows']}")
    
    print("\n🎯 Optuna优化结果:")
    for timeframe, results in optuna_results.items():
        print(f"  {timeframe}: 最佳AUC={results['best_auc']:.4f}, 试验次数={results['n_trials']}")
    
    print("\n💰 回测结果:")
    for timeframe, results in backtest_results.items():
        print(f"  {timeframe}: 交易数={results['total_trades']}, 胜率={results['win_rate']:.2%}, "
              f"总收益={results['total_returns']:.4f}, 平均收益={results['avg_returns']:.4f}")
    
    print("\n🎯 特征筛选结果:")
    for timeframe, results in feature_results.items():
        print(f"  {timeframe}: 特征数={results['total_features']}, 减少={results['reduction']}个")
    
    print("\n🎯 决策建议:")
    print("=" * 50)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # 保存分析结果
    analysis_results = {
        'walk_forward_results': walk_forward_results,
        'optuna_results': optuna_results,
        'backtest_results': backtest_results,
        'feature_results': feature_results,
        'recommendations': recommendations
    }
    
    with open('sprint_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n✅ 分析结果已保存: sprint_analysis_results.json")

if __name__ == "__main__":
    main() 