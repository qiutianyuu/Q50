#!/usr/bin/env python3
"""
标签重新设计 - 尝试不同的标签定义
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置数据路径
DATA_DIR = Path("data")
PROCESSED_DIR = Path("/Users/qiutianyu/data/processed")

def test_different_labels():
    """测试不同的标签定义"""
    print("🏷️ 测试不同的标签定义...")
    
    # 尝试不同的数据文件路径
    possible_files = [
        PROCESSED_DIR / "merged_15m_2023_2025.parquet",
        DATA_DIR / "merged_15m_2023_2025.parquet",
        Path("merged_15m_2023_2025.parquet")
    ]
    
    df = None
    for file_path in possible_files:
        if file_path.exists():
            print(f"📁 使用数据文件: {file_path}")
            df = pd.read_parquet(file_path)
            break
    
    if df is None:
        print("❌ 找不到K线数据文件，请确保数据文件存在")
        return []
    
    # 测试不同的标签定义
    label_configs = [
        {
            'name': '未来6根累计收益±0.15%',
            'horizon': 6,
            'pos_thr': 0.0015,
            'neg_thr': -0.0015
        },
        {
            'name': '未来12根累计收益±0.2%',
            'horizon': 12,
            'pos_thr': 0.002,
            'neg_thr': -0.002
        },
        {
            'name': '未来24根累计收益±0.3%',
            'horizon': 24,
            'pos_thr': 0.003,
            'neg_thr': -0.003
        },
        {
            'name': '未来48根累计收益±0.5%',
            'horizon': 48,
            'pos_thr': 0.005,
            'neg_thr': -0.005
        },
        {
            'name': '简单下一根涨跌',
            'horizon': 1,
            'pos_thr': 0.0001,
            'neg_thr': -0.0001
        }
    ]
    
    results = []
    
    for config in label_configs:
        print(f"\n🔧 测试: {config['name']}")
        
        # 生成标签
        if config['name'] == '简单下一根涨跌':
            # 简单的下一根涨跌
            labels = (df['close'].shift(-1) > df['close']).astype(int)
            labels = labels.fillna(0)  # 最后一行填0
        else:
            # 未来N根累计收益阈值
            future_returns = df['close'].pct_change(config['horizon']).shift(-config['horizon'])
            labels = np.where(future_returns > config['pos_thr'], 1, 
                             np.where(future_returns < config['neg_thr'], 0, -1))
            labels = pd.Series(labels).fillna(-1)
        
        # 统计标签分布
        total_samples = len(labels)
        long_signals = (labels == 1).sum()
        short_signals = (labels == 0).sum()
        no_trade = (labels == -1).sum()
        
        print(f"总样本: {total_samples}")
        print(f"做多信号: {long_signals} ({long_signals/total_samples*100:.1f}%)")
        print(f"做空信号: {short_signals} ({short_signals/total_samples*100:.1f}%)")
        print(f"不交易: {no_trade} ({no_trade/total_samples*100:.1f}%)")
        print(f"交易信号占比: {(long_signals+short_signals)/total_samples*100:.1f}%")
        
        # 只保留交易信号
        trade_mask = labels != -1
        if trade_mask.sum() < 1000:
            print("⚠️ 交易信号太少，跳过")
            continue
        
        # 创建简化的特征（只使用技术指标）
        tech_features = ['rsi_14', 'macd_diff', 'bb_percent', 'stoch_k', 'adx_14', 'atr_norm']
        
        # 计算技术指标
        from ta.trend import ADXIndicator, MACD
        from ta.momentum import RSIIndicator, StochasticOscillator
        from ta.volatility import BollingerBands, AverageTrueRange
        
        hi, lo, close = df["high"], df["low"], df["close"]
        
        df_features = pd.DataFrame()
        df_features["rsi_14"] = RSIIndicator(close, window=14).rsi()
        df_features["macd_diff"] = MACD(close).macd_diff()
        
        bb = BollingerBands(close, window=20, window_dev=2)
        df_features["bb_percent"] = bb.bollinger_pband()
        
        stoch = StochasticOscillator(hi, lo, close, window=14, smooth_window=3)
        df_features["stoch_k"] = stoch.stoch()
        
        df_features["adx_14"] = ADXIndicator(hi, lo, close, window=14).adx()
        df_features["atr_norm"] = AverageTrueRange(hi, lo, close, window=14).average_true_range() / close
        
        # 准备数据
        X = df_features.fillna(0)
        y = labels[trade_mask]
        X = X[trade_mask]
        
        # 时间排序 —— 保留原始索引，再统一 reset
        sorted_idx = df.loc[trade_mask].sort_values('timestamp').index  # 原始索引
        X_sorted = X.loc[sorted_idx].reset_index(drop=True)
        y_sorted = y.loc[sorted_idx].reset_index(drop=True)
        # 安全检查
        assert len(X_sorted) == len(y_sorted), "索引对齐失败，特征与标签行数不一致"
        
        # 分割数据
        split_idx = int(len(X_sorted) * 0.8)
        X_train = X_sorted[:split_idx]
        y_train = y_sorted[:split_idx]
        X_test = X_sorted[split_idx:]
        y_test = y_sorted[split_idx:]
        
        if len(X_train) < 1000 or len(X_test) < 100:
            print("⚠️ 样本不足，跳过")
            continue
        
        # 训练简单模型
        model = xgb.XGBClassifier(
            max_depth=3, n_estimators=100, learning_rate=0.1,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42
        )
        model.fit(X_train, y_train, verbose=0)
        
        # 评估
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        overfitting = train_auc - test_auc
        
        result = {
            'name': config['name'],
            'horizon': config.get('horizon', 1),
            'pos_thr': config.get('pos_thr', 0),
            'neg_thr': config.get('neg_thr', 0),
            'trade_signals': long_signals + short_signals,
            'signal_ratio': (long_signals + short_signals) / total_samples,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfitting': overfitting
        }
        
        results.append(result)
        
        print(f"训练AUC: {train_auc:.4f}")
        print(f"测试AUC: {test_auc:.4f}")
        print(f"过拟合程度: {overfitting:.4f}")
    
    # 总结
    print(f"\n📊 标签定义测试总结:")
    for result in results:
        status = "✅" if result['overfitting'] < 0.05 and result['test_auc'] > 0.55 else "⚠️"
        print(f"{status} {result['name']}: 测试AUC={result['test_auc']:.4f}, 过拟合={result['overfitting']:.4f}, 信号占比={result['signal_ratio']:.1%}")
    
    return results

def main():
    print("🏷️ 标签重新设计测试")
    
    results = test_different_labels()
    
    if results:
        # 找出最佳标签定义
        best_result = max(results, key=lambda x: x['test_auc'])
        print(f"\n🎯 最佳标签定义: {best_result['name']}")
        print(f"测试AUC: {best_result['test_auc']:.4f}")
        print(f"过拟合程度: {best_result['overfitting']:.4f}")
        print(f"信号占比: {best_result['signal_ratio']:.1%}")
        
        # 生成最佳标签
        if best_result['name'] != '简单下一根涨跌':
            print(f"\n📝 生成最佳标签文件...")
            
            # 重新读取数据
            possible_files = [
                PROCESSED_DIR / "merged_15m_2023_2025.parquet",
                DATA_DIR / "merged_15m_2023_2025.parquet",
                Path("merged_15m_2023_2025.parquet")
            ]
            
            df = None
            for file_path in possible_files:
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    break
            
            if df is not None:
                future_returns = df['close'].pct_change(best_result['horizon']).shift(-best_result['horizon'])
                labels = np.where(future_returns > best_result['pos_thr'], 1, 
                                 np.where(future_returns < best_result['neg_thr'], 0, -1))
                
                label_df = pd.DataFrame({
                    'timestamp': df['timestamp'],
                    'close': df['close'],
                    'future_return': future_returns,
                    'label': labels
                })
                
                output_file = DATA_DIR / "label_15m_best.csv"
                label_df.to_csv(output_file, index=False)
                print(f"✅ 最佳标签已保存: {output_file}")

if __name__ == "__main__":
    main() 