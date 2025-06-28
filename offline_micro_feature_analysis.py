#!/usr/bin/env python3
"""
离线微观特征分析脚本
批量处理WebSocket数据，计算信息系数，训练简单模型验证alpha
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OfflineMicroFeatureAnalysis:
    def __init__(self, symbol: str = "ETH-USDT", window_size: int = 50):
        self.symbol = symbol
        self.window_size = window_size
        self.websocket_dir = "data/websocket"
        self.output_dir = "data/analysis"
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_all_websocket_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载所有WebSocket数据"""
        logger.info("📊 加载所有WebSocket数据...")
        
        # 加载所有OrderBook数据
        orderbook_files = glob.glob(f"{self.websocket_dir}/orderbook_{self.symbol}_*.parquet")
        orderbook_files.sort()
        
        orderbook_dfs = []
        for file in orderbook_files:
            df = pd.read_parquet(file)
            if 'ts' in df.columns:
                df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
            orderbook_dfs.append(df)
        
        orderbook_df = pd.concat(orderbook_dfs, ignore_index=True)
        orderbook_df = orderbook_df.sort_values('timestamp').reset_index(drop=True)
        
        # 加载所有Trades数据
        trades_files = glob.glob(f"{self.websocket_dir}/trades_{self.symbol}_*.parquet")
        trades_files.sort()
        
        trades_dfs = []
        for file in trades_files:
            df = pd.read_parquet(file)
            if 'ts' in df.columns:
                df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
            trades_dfs.append(df)
        
        trades_df = pd.concat(trades_dfs, ignore_index=True)
        trades_df = trades_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"📈 OrderBook数据: {len(orderbook_df)} 条")
        logger.info(f"📊 Trades数据: {len(trades_df)} 条")
        
        return orderbook_df, trades_df
    
    def calculate_micro_price_features_batch(self, orderbook_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算微价格特征"""
        logger.info("📈 计算微价格特征...")
        
        features = []
        
        for _, row in orderbook_df.iterrows():
            # 直接使用解析好的OrderBook数据
            bid_prices = []
            bid_sizes = []
            ask_prices = []
            ask_sizes = []
            
            # 提取5档买卖盘数据
            for i in range(1, 6):
                bid_price_col = f'bid{i}_price'
                bid_size_col = f'bid{i}_size'
                ask_price_col = f'ask{i}_price'
                ask_size_col = f'ask{i}_size'
                
                if bid_price_col in row and bid_size_col in row:
                    bid_prices.append(row[bid_price_col])
                    bid_sizes.append(row[bid_size_col])
                
                if ask_price_col in row and ask_size_col in row:
                    ask_prices.append(row[ask_price_col])
                    ask_sizes.append(row[ask_size_col])
            
            if not bid_prices or not ask_prices:
                continue
            
            # 加权平均价格 (VWAP)
            bid_vwap = np.average(bid_prices, weights=bid_sizes) if bid_sizes else 0
            ask_vwap = np.average(ask_prices, weights=ask_sizes) if ask_sizes else 0
            
            # 中间价
            mid_price = (bid_prices[0] + ask_prices[0]) / 2 if bid_prices and ask_prices else 0
            
            # 买卖价差
            spread = ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else 0
            spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
            
            # 订单簿不平衡
            bid_volume = sum(bid_sizes)
            ask_volume = sum(ask_sizes)
            volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            # 价格压力
            price_pressure = (ask_vwap - mid_price) / mid_price if mid_price > 0 else 0
            
            features.append({
                'timestamp': row.get('timestamp', pd.Timestamp.now()),
                'mid_price': mid_price,
                'bid_vwap': bid_vwap,
                'ask_vwap': ask_vwap,
                'spread': spread,
                'spread_bps': spread_bps,
                'volume_imbalance': volume_imbalance,
                'price_pressure': price_pressure,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': bid_volume + ask_volume
            })
        
        return pd.DataFrame(features)
    
    def calculate_order_flow_features_batch(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算订单流特征"""
        logger.info("📊 计算订单流特征...")
        
        features = []
        
        # 按时间窗口分组计算特征
        trades_df = trades_df.sort_values('timestamp')
        
        for i in range(0, len(trades_df), self.window_size):
            window_trades = trades_df.iloc[i:i+self.window_size]
            
            if len(window_trades) == 0:
                continue
            
            # 计算基础统计
            prices = window_trades['price'].astype(float)
            sizes = window_trades['size'].astype(float)
            sides = window_trades['side'].astype(str)
            
            # 价格特征
            price_mean = prices.mean()
            price_std = prices.std()
            price_range = prices.max() - prices.min()
            
            # 成交量特征
            total_volume = sizes.sum()
            avg_trade_size = sizes.mean()
            large_trades = (sizes > sizes.quantile(0.9)).sum()
            
            # 买卖压力
            buy_volume = sizes[sides == 'buy'].sum()
            sell_volume = sizes[sides == 'sell'].sum()
            buy_ratio = buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5
            
            # 价格动量
            if len(prices) > 1:
                price_momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            else:
                price_momentum = 0
            
            # 成交量加权平均价格
            vwap = np.average(prices, weights=sizes)
            
            # 时间特征
            time_span = (window_trades['timestamp'].max() - window_trades['timestamp'].min()).total_seconds()
            trade_frequency = len(window_trades) / time_span if time_span > 0 else 0
            
            features.append({
                'timestamp': window_trades['timestamp'].iloc[-1],
                'price_mean': price_mean,
                'price_std': price_std,
                'price_range': price_range,
                'total_volume': total_volume,
                'avg_trade_size': avg_trade_size,
                'large_trades': large_trades,
                'buy_ratio': buy_ratio,
                'price_momentum': price_momentum,
                'vwap': vwap,
                'trade_frequency': trade_frequency,
                'trade_count': len(window_trades)
            })
        
        return pd.DataFrame(features)
    
    def generate_labels(self, features_df: pd.DataFrame, forward_period: int = 5) -> pd.DataFrame:
        """生成标签：未来价格变化"""
        logger.info(f"🏷️ 生成标签 (前向{forward_period}期)...")
        
        # 计算未来价格变化
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
        
        # 未来价格变化
        features_df['future_price_change'] = features_df['mid_price'].shift(-forward_period) / features_df['mid_price'] - 1
        
        # 二分类标签
        features_df['label_binary'] = (features_df['future_price_change'] > 0).astype(int)
        
        # 三分类标签 (上涨/下跌/震荡) - 先不astype(int)
        features_df['label_three'] = pd.cut(
            features_df['future_price_change'], 
            bins=[-np.inf, -0.001, 0.001, np.inf], 
            labels=[0, 1, 2]
        )
        
        # 移除最后几行（没有未来数据）
        features_df = features_df.dropna(subset=['future_price_change', 'label_three']).reset_index(drop=True)
        features_df['label_three'] = features_df['label_three'].astype(int)
        
        logger.info(f"📊 标签统计:")
        logger.info(f"  二分类: {features_df['label_binary'].value_counts().to_dict()}")
        logger.info(f"  三分类: {features_df['label_three'].value_counts().to_dict()}")
        logger.info(f"  价格变化均值: {features_df['future_price_change'].mean():.6f}")
        logger.info(f"  价格变化标准差: {features_df['future_price_change'].std():.6f}")
        
        return features_df
    
    def calculate_information_coefficient(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """计算信息系数 (IC)"""
        logger.info("📊 计算信息系数...")
        
        # 选择数值型特征
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f not in [
            'future_price_change', 'label_binary', 'label_three'
        ]]
        
        ic_results = []
        
        for feature in numeric_features:
            # 移除缺失值
            valid_data = features_df[[feature, 'future_price_change']].dropna()
            
            if len(valid_data) < 10:  # 至少需要10个有效数据点
                continue
            
            # 计算皮尔逊相关系数
            pearson_corr, pearson_p = stats.pearsonr(valid_data[feature], valid_data['future_price_change'])
            
            # 计算斯皮尔曼相关系数
            spearman_corr, spearman_p = stats.spearmanr(valid_data[feature], valid_data['future_price_change'])
            
            ic_results.append({
                'feature': feature,
                'pearson_ic': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_ic': spearman_corr,
                'spearman_p': spearman_p,
                'abs_pearson_ic': abs(pearson_corr),
                'abs_spearman_ic': abs(spearman_corr),
                'sample_size': len(valid_data)
            })
        
        ic_df = pd.DataFrame(ic_results)
        ic_df = ic_df.sort_values('abs_pearson_ic', ascending=False)
        
        logger.info(f"📈 IC分析完成，共{len(ic_df)}个特征")
        logger.info(f"🏆 前5个最强特征:")
        for i, row in ic_df.head().iterrows():
            logger.info(f"  {row['feature']}: Pearson={row['pearson_ic']:.4f}, Spearman={row['spearman_ic']:.4f}")
        
        return ic_df
    
    def train_simple_model(self, features_df: pd.DataFrame, top_features: int = 10) -> Dict:
        """训练简单模型"""
        logger.info(f"🤖 训练简单模型 (使用前{top_features}个特征)...")
        
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import StandardScaler
            
            # 获取IC最高的特征
            ic_df = self.calculate_information_coefficient(features_df)
            top_feature_names = ic_df.head(top_features)['feature'].tolist()
            
            # 准备数据
            X = features_df[top_feature_names].fillna(0)
            y_binary = features_df['label_binary']
            y_three = features_df['label_three']
            
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 二分类模型
            rf_binary = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_scores_binary = cross_val_score(rf_binary, X_scaled, y_binary, cv=5, scoring='accuracy')
            
            # 三分类模型
            rf_three = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_scores_three = cross_val_score(rf_three, X_scaled, y_three, cv=5, scoring='accuracy')
            
            # 特征重要性
            rf_binary.fit(X_scaled, y_binary)
            feature_importance = pd.DataFrame({
                'feature': top_feature_names,
                'importance': rf_binary.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results = {
                'binary_accuracy_mean': cv_scores_binary.mean(),
                'binary_accuracy_std': cv_scores_binary.std(),
                'three_accuracy_mean': cv_scores_three.mean(),
                'three_accuracy_std': cv_scores_three.std(),
                'feature_importance': feature_importance,
                'top_features': top_feature_names,
                'cv_scores_binary': cv_scores_binary,
                'cv_scores_three': cv_scores_three
            }
            
            logger.info(f"📊 模型结果:")
            logger.info(f"  二分类准确率: {results['binary_accuracy_mean']:.4f} ± {results['binary_accuracy_std']:.4f}")
            logger.info(f"  三分类准确率: {results['three_accuracy_mean']:.4f} ± {results['three_accuracy_std']:.4f}")
            logger.info(f"🏆 前5个重要特征:")
            for i, row in feature_importance.head().iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return results
            
        except ImportError:
            logger.warning("❌ sklearn未安装，跳过模型训练")
            return {}
    
    def save_results(self, features_df: pd.DataFrame, ic_df: pd.DataFrame, model_results: Dict):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存特征数据
        features_file = f"{self.output_dir}/micro_features_{self.symbol}_{timestamp}.parquet"
        features_df.to_parquet(features_file, index=False)
        logger.info(f"💾 保存特征数据: {features_file}")
        
        # 保存IC结果
        ic_file = f"{self.output_dir}/ic_analysis_{self.symbol}_{timestamp}.csv"
        ic_df.to_csv(ic_file, index=False)
        logger.info(f"💾 保存IC分析: {ic_file}")
        
        # 保存模型结果
        if model_results:
            model_file = f"{self.output_dir}/model_results_{self.symbol}_{timestamp}.json"
            import json
            # 转换numpy数组为list以便JSON序列化
            model_results_json = {
                'binary_accuracy_mean': float(model_results['binary_accuracy_mean']),
                'binary_accuracy_std': float(model_results['binary_accuracy_std']),
                'three_accuracy_mean': float(model_results['three_accuracy_mean']),
                'three_accuracy_std': float(model_results['three_accuracy_std']),
                'top_features': model_results['top_features'],
                'cv_scores_binary': model_results['cv_scores_binary'].tolist(),
                'cv_scores_three': model_results['cv_scores_three'].tolist(),
                'feature_importance': model_results['feature_importance'].to_dict('records')
            }
            with open(model_file, 'w') as f:
                json.dump(model_results_json, f, indent=2)
            logger.info(f"💾 保存模型结果: {model_file}")
    
    def run_analysis(self, forward_period: int = 5, top_features: int = 10) -> Dict:
        """运行完整分析流程"""
        logger.info("🚀 开始离线微观特征分析...")
        
        # 1. 加载数据
        orderbook_df, trades_df = self.load_all_websocket_data()
        
        # 2. 计算特征
        micro_price_features = self.calculate_micro_price_features_batch(orderbook_df)
        order_flow_features = self.calculate_order_flow_features_batch(trades_df)
        
        # 3. 合并特征
        features_df = pd.merge_asof(
            micro_price_features.sort_values('timestamp'),
            order_flow_features.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            suffixes=('', '_flow')
        )
        
        # 4. 计算衍生特征
        features_df['price_change'] = features_df['mid_price'].pct_change()
        features_df['price_change_abs'] = features_df['price_change'].abs()
        features_df['volume_change'] = features_df['total_volume'].pct_change()
        features_df['buy_pressure_change'] = features_df['buy_ratio'].diff()
        features_df['spread_change'] = features_df['spread'].pct_change()
        features_df['price_volatility'] = features_df['price_change'].rolling(5).std()
        features_df['price_trend'] = features_df['mid_price'].rolling(10).mean()
        features_df['trend_deviation'] = (features_df['mid_price'] - features_df['price_trend']) / features_df['price_trend']
        
        logger.info(f"📊 合并后特征数据: {len(features_df)} 条, {len(features_df.columns)} 列")
        
        # 5. 生成标签
        features_df = self.generate_labels(features_df, forward_period)
        
        # 6. 计算IC
        ic_df = self.calculate_information_coefficient(features_df)
        
        # 7. 训练模型
        model_results = self.train_simple_model(features_df, top_features)
        
        # 8. 保存结果
        self.save_results(features_df, ic_df, model_results)
        
        logger.info("✅ 离线分析完成！")
        
        return {
            'features_df': features_df,
            'ic_df': ic_df,
            'model_results': model_results
        }

def main():
    parser = argparse.ArgumentParser(description="离线微观特征分析")
    parser.add_argument("--symbol", default="ETH-USDT", help="交易对")
    parser.add_argument("--window-size", type=int, default=50, help="滑动窗口大小")
    parser.add_argument("--forward-period", type=int, default=5, help="前向预测期数")
    parser.add_argument("--top-features", type=int, default=10, help="使用前N个特征训练模型")
    
    args = parser.parse_args()
    
    # 创建分析实例
    analyzer = OfflineMicroFeatureAnalysis(
        symbol=args.symbol,
        window_size=args.window_size
    )
    
    # 运行分析
    results = analyzer.run_analysis(
        forward_period=args.forward_period,
        top_features=args.top_features
    )
    
    # 打印总结
    print("\n" + "="*50)
    print("📊 分析总结")
    print("="*50)
    print(f"数据量: {len(results['features_df'])} 条")
    print(f"特征数: {len(results['features_df'].columns)} 个")
    
    if results['model_results']:
        print(f"二分类准确率: {results['model_results']['binary_accuracy_mean']:.4f}")
        print(f"三分类准确率: {results['model_results']['three_accuracy_mean']:.4f}")
    
    print("="*50)

if __name__ == "__main__":
    main() 