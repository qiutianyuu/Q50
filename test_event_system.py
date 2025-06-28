#!/usr/bin/env python3
"""
事件系统测试脚本 - 验证事件检测和标签生成功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def test_event_detection():
    """测试事件检测功能"""
    print("🔍 测试事件检测系统...")
    
    # 检查输入文件
    input_file = "data/features_15m_enhanced.parquet"
    if not Path(input_file).exists():
        print(f"⚠️ 输入文件不存在: {input_file}")
        print("请先运行特征工程脚本生成特征文件")
        return False
    
    # 运行事件检测
    output_file = "data/events_15m.parquet"
    cmd = [
        sys.executable, "detect_events.py",
        "--input", input_file,
        "--output", output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ 事件检测成功!")
        print(result.stdout)
        
        # 验证输出文件
        if Path(output_file).exists():
            df = pd.read_parquet(output_file)
            print(f"📊 事件文件统计:")
            print(f"  样本数: {len(df):,}")
            print(f"  特征数: {len(df.columns)}")
            
            # 检查事件特征
            event_features = [col for col in df.columns if any(event_type in col for event_type in 
                            ['breakout', 'reversal', 'spike', 'cross', 'oversold', 'overbought', 'whale_'])]
            print(f"  事件特征数: {len(event_features)}")
            
            # 检查聚合特征
            if 'event_strength' in df.columns:
                print(f"  事件强度范围: {df['event_strength'].min():.3f} 到 {df['event_strength'].max():.3f}")
            if 'event_density' in df.columns:
                print(f"  事件密度范围: {df['event_density'].min():.0f} 到 {df['event_density'].max():.0f}")
            
            return True
        else:
            print("❌ 事件检测输出文件不存在")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 事件检测失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def test_label_generation():
    """测试标签生成功能"""
    print("\n🏷️ 测试标签生成系统...")
    
    # 检查输入文件
    input_file = "data/events_15m.parquet"
    if not Path(input_file).exists():
        print(f"⚠️ 事件文件不存在: {input_file}")
        print("请先运行事件检测脚本")
        return False
    
    # 测试不同的标签策略
    strategies = ['event_strength', 'event_combination', 'event_sequential']
    
    for strategy in strategies:
        print(f"\n🎯 测试策略: {strategy}")
        
        output_file = f"data/labels_15m_{strategy}.parquet"
        cmd = [
            sys.executable, "label_events.py",
            "--input", input_file,
            "--output", output_file,
            "--strategy", strategy,
            "--min_strength", "0.2",
            "--max_strength", "0.9",
            "--min_density", "2",
            "--hold_period", "4",
            "--min_profit", "0.0005",
            "--max_loss", "-0.0015"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {strategy} 标签生成成功!")
            
            # 验证输出文件
            if Path(output_file).exists():
                df = pd.read_parquet(output_file)
                print(f"📊 {strategy} 标签统计:")
                print(f"  样本数: {len(df):,}")
                
                if 'label' in df.columns:
                    label_counts = df['label'].value_counts()
                    print(f"  标签分布: {label_counts.to_dict()}")
                    
                    # 计算交易信号比例
                    trade_signals = (df['label'] != -1).sum()
                    trade_ratio = trade_signals / len(df) * 100
                    print(f"  交易信号比例: {trade_ratio:.1f}%")
                    
                    # 分析收益
                    if 'net_return' in df.columns:
                        trade_mask = df['label'] != -1
                        if trade_mask.sum() > 0:
                            trade_returns = df.loc[trade_mask, 'net_return']
                            print(f"  平均净收益: {trade_returns.mean():.6f}")
                            print(f"  正收益比例: {(trade_returns > 0).sum() / len(trade_returns)*100:.1f}%")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {strategy} 标签生成失败: {e}")
            print(f"错误输出: {e.stderr}")
            continue
    
    return True

def test_integration():
    """测试完整流程集成"""
    print("\n🔄 测试完整流程集成...")
    
    # 检查所有必要的文件
    required_files = [
        "data/features_15m_enhanced.parquet",
        "data/events_15m.parquet",
        "data/labels_15m_event_strength.parquet"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"⚠️ 缺少文件: {missing_files}")
        return False
    
    # 加载数据并验证一致性
    try:
        features_df = pd.read_parquet("data/features_15m_enhanced.parquet")
        events_df = pd.read_parquet("data/events_15m.parquet")
        labels_df = pd.read_parquet("data/labels_15m_event_strength.parquet")
        
        print("📊 数据一致性检查:")
        print(f"  特征数据样本数: {len(features_df):,}")
        print(f"  事件数据样本数: {len(events_df):,}")
        print(f"  标签数据样本数: {len(labels_df):,}")
        
        # 检查时间戳一致性
        if len(features_df) == len(events_df) == len(labels_df):
            print("✅ 数据长度一致")
        else:
            print("❌ 数据长度不一致")
            return False
        
        # 检查时间戳对齐
        features_timestamps = set(features_df['timestamp'])
        events_timestamps = set(events_df['timestamp'])
        labels_timestamps = set(labels_df['timestamp'])
        
        if features_timestamps == events_timestamps == labels_timestamps:
            print("✅ 时间戳对齐")
        else:
            print("❌ 时间戳不对齐")
            return False
        
        # 验证事件特征
        event_features = [col for col in events_df.columns if any(event_type in col for event_type in 
                        ['breakout', 'reversal', 'spike', 'cross', 'oversold', 'overbought', 'whale_'])]
        print(f"  检测到的事件特征: {len(event_features)}")
        
        # 验证标签质量
        if 'label' in labels_df.columns:
            trade_signals = (labels_df['label'] != -1).sum()
            print(f"  生成的交易信号: {trade_signals:,}")
            
            if trade_signals > 0:
                print("✅ 成功生成交易信号")
            else:
                print("⚠️ 未生成交易信号，可能需要调整参数")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def generate_test_report():
    """生成测试报告"""
    print("\n📋 生成测试报告...")
    
    report = {
        "test_time": datetime.now().isoformat(),
        "event_detection": False,
        "label_generation": False,
        "integration": False,
        "summary": []
    }
    
    # 运行测试
    if test_event_detection():
        report["event_detection"] = True
        report["summary"].append("✅ 事件检测测试通过")
    else:
        report["summary"].append("❌ 事件检测测试失败")
    
    if test_label_generation():
        report["label_generation"] = True
        report["summary"].append("✅ 标签生成测试通过")
    else:
        report["summary"].append("❌ 标签生成测试失败")
    
    if test_integration():
        report["integration"] = True
        report["summary"].append("✅ 集成测试通过")
    else:
        report["summary"].append("❌ 集成测试失败")
    
    # 保存报告
    report_file = "test_report.json"
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📁 测试报告已保存: {report_file}")
    
    # 打印总结
    print("\n🎯 测试总结:")
    for summary in report["summary"]:
        print(f"  {summary}")
    
    if all([report["event_detection"], report["label_generation"], report["integration"]]):
        print("\n🎉 所有测试通过! 事件系统准备就绪")
    else:
        print("\n⚠️ 部分测试失败，请检查相关脚本和配置")

def main():
    print("🧪 RexKing 事件系统测试")
    print("=" * 50)
    
    # 检查必要文件
    required_scripts = ["detect_events.py", "label_events.py"]
    missing_scripts = [s for s in required_scripts if not Path(s).exists()]
    
    if missing_scripts:
        print(f"❌ 缺少必要脚本: {missing_scripts}")
        return
    
    print("✅ 所有必要脚本存在")
    
    # 运行测试
    generate_test_report()

if __name__ == "__main__":
    main() 