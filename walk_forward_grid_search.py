import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
import warnings
import random
warnings.filterwarnings('ignore')

# 导入策略类
from rexking_eth_8_2_strategy import RexKingETH82Strategy

# 参数网格
PARAM_GRID = {
    'atr_threshold': [0.003, 0.005, 0.008],  # ATR阈值
    'bb_threshold': [2, 3, 4],  # BB阈值
    'funding_threshold': [0.00003, 0.00005, 0.00008],  # Funding阈值
    'volume_threshold': [0.1, 0.15, 0.2],  # 15m成交量阈值
    'sl_mult': [0.3, 0.5, 0.8],  # 止损倍数
    'tp_mult': [3.0, 5.0, 8.0],  # 止盈倍数
}

INIT_CAP = 7000.0
FEE_RATE = 0.00075
MIN_POSITION = 28
MAX_POSITION = 1400

class WalkForwardGridSearch:
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.results = []
        
        # 简化的参数网格
        self.param_grid = {
            'volume_multiplier': [1.1, 1.2, 1.3],
            'rsi_bull': [48, 50, 52],
            'rsi_range': [43, 45, 47],
            'atr_threshold': [0.0025, 0.003, 0.0035],
            'signal_threshold': [0.04, 0.05, 0.06]
        }
    
    def load_data(self, file_path):
        """加载数据"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, header=None)
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        df['timestamp'] = pd.to_datetime(df['open_time'] // 1000, unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        
        df['returns'] = df['close'].pct_change()
        df['high_low'] = df['high'] - df['low']
        df['close_open'] = df['close'] - df['open']
        
        return df
    
    def split_data(self, df, train_days=15, test_days=5):
        """分割训练和测试数据"""
        splits = []
        current_start = df['timestamp'].min()
        
        while current_start < df['timestamp'].max():
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            
            if test_end > df['timestamp'].max():
                break
            
            train_data = df[(df['timestamp'] >= current_start) & (df['timestamp'] < train_end)].copy()
            test_data = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)].copy()
            
            if len(train_data) > 100 and len(test_data) > 20:
                splits.append({
                    'train': train_data,
                    'test': test_data,
                    'train_start': current_start,
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': test_end
                })
            
            current_start = train_end
        
        return splits
    
    def evaluate_strategy(self, df, params):
        """评估单个参数组合"""
        try:
            # 创建策略实例
            strategy = RexKingETH82Strategy(initial_capital=self.initial_capital)
            
            # 临时修改策略参数
            original_volume_spike = None
            original_rsi_thresholds = None
            original_atr_threshold = None
            original_signal_threshold = None
            
            # 保存原始参数
            if hasattr(strategy, 'volume_multiplier'):
                original_volume_spike = strategy.volume_multiplier
            if hasattr(strategy, 'rsi_bull_threshold'):
                original_rsi_thresholds = (strategy.rsi_bull_threshold, strategy.rsi_range_threshold)
            if hasattr(strategy, 'atr_threshold'):
                original_atr_threshold = strategy.atr_threshold
            if hasattr(strategy, 'signal_strength_threshold'):
                original_signal_threshold = strategy.signal_strength_threshold
            
            # 应用新参数
            strategy.volume_multiplier = params['volume_multiplier']
            strategy.rsi_bull_threshold = params['rsi_bull']
            strategy.rsi_range_threshold = params['rsi_range']
            strategy.atr_threshold = params['atr_threshold']
            strategy.signal_strength_threshold = params['signal_threshold']
            
            # 计算指标
            df_with_indicators = strategy.calculate_indicators(df.copy())
            
            # 回测
            strategy.backtest(df_with_indicators)
            
            # 分析结果
            if not strategy.trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'annualized_return': 0,
                    'max_drawdown': 0,
                    'profit_factor': 0,
                    'trades_per_day': 0
                }
            
            trades_df = pd.DataFrame(strategy.trades)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_return = (strategy.capital / self.initial_capital - 1) * 100
            
            # 年化收益率
            if len(trades_df) > 1:
                start_time = trades_df['entry_time'].min()
                end_time = trades_df['exit_time'].max()
                days = (end_time - start_time).days
                annual_return = (strategy.capital / self.initial_capital - 1) * (365 / days) * 100 if days > 0 else 0
            else:
                annual_return = 0
            
            # 盈亏比
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0
            profit_factor = avg_win * winning_trades / (avg_loss * (total_trades - winning_trades)) if (total_trades - winning_trades) > 0 else float('inf')
            
            # 信号密度
            trades_per_day = total_trades / max(1, (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days)
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate * 100,
                'total_return': total_return,
                'annualized_return': annual_return,
                'max_drawdown': strategy.max_drawdown * 100,
                'profit_factor': profit_factor,
                'trades_per_day': trades_per_day
            }
            
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return None
    
    def run_grid_search(self, data_path, max_combinations=20):
        """运行网格搜索"""
        print("Starting Walk-Forward Grid Search...")
        
        # 加载数据
        df = self.load_data(data_path)
        
        # 分割数据
        splits = self.split_data(df)
        print(f"Created {len(splits)} train/test splits")
        
        # 生成参数组合
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # 限制组合数量
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > max_combinations:
            # 随机采样
            np.random.seed(42)
            selected_indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations = [all_combinations[i] for i in selected_indices]
        else:
            combinations = all_combinations
        
        print(f"Testing {len(combinations)} parameter combinations")
        
        # 对每个分割进行网格搜索
        for split_idx, split in enumerate(splits):
            print(f"\nProcessing split {split_idx + 1}/{len(splits)}")
            print(f"Train: {split['train_start'].date()} to {split['train_end'].date()}")
            print(f"Test:  {split['test_start'].date()} to {split['test_end'].date()}")
            
            # 在训练集上找到最佳参数
            best_params = None
            best_score = -float('inf')
            
            for i, combination in enumerate(combinations):
                if i % 5 == 0:
                    print(f"  Progress: {i}/{len(combinations)}")
                
                params = dict(zip(param_names, combination))
                
                # 在训练集上评估
                train_result = self.evaluate_strategy(split['train'], params)
                
                if train_result and train_result['total_trades'] > 0:
                    # 计算综合得分
                    score = (
                        train_result['annualized_return'] * 0.4 +
                        train_result['win_rate'] * 0.2 +
                        train_result['profit_factor'] * 0.2 +
                        (1 / (1 + train_result['max_drawdown'])) * 0.1 +
                        min(train_result['trades_per_day'] / 10, 1) * 0.1
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
            
            if best_params:
                # 在测试集上验证最佳参数
                test_result = self.evaluate_strategy(split['test'], best_params)
                
                if test_result:
                    self.results.append({
                        'split_idx': split_idx,
                        'train_start': split['train_start'],
                        'train_end': split['train_end'],
                        'test_start': split['test_start'],
                        'test_end': split['test_end'],
                        'best_params': best_params,
                        'test_result': test_result,
                        'best_score': best_score
                    })
                    
                    print(f"  Best params: {best_params}")
                    print(f"  Test:  {test_result['annualized_return']:.1f}% AR, {test_result['win_rate']:.1f}% WR")
        
        return self.results
    
    def analyze_results(self):
        """分析网格搜索结果"""
        if not self.results:
            print("No results to analyze!")
            return
        
        print("\n" + "="*80)
        print("WALK-FORWARD GRID SEARCH RESULTS")
        print("="*80)
        
        # 统计测试集表现
        test_returns = [r['test_result']['annualized_return'] for r in self.results]
        test_win_rates = [r['test_result']['win_rate'] for r in self.results]
        test_profit_factors = [r['test_result']['profit_factor'] for r in self.results]
        test_drawdowns = [r['test_result']['max_drawdown'] for r in self.results]
        
        print(f"Number of splits: {len(self.results)}")
        print(f"Average Test Return: {np.mean(test_returns):.2f}%")
        print(f"Average Test Win Rate: {np.mean(test_win_rates):.1f}%")
        print(f"Average Test Profit Factor: {np.mean(test_profit_factors):.2f}")
        print(f"Average Test Max Drawdown: {np.mean(test_drawdowns):.2f}%")
        
        # 稳定性分析
        positive_returns = sum(1 for r in test_returns if r > 0)
        print(f"Positive Return Rate: {positive_returns/len(test_returns)*100:.1f}%")
        
        # 参数频率分析
        param_frequency = {}
        for result in self.results:
            for param, value in result['best_params'].items():
                if param not in param_frequency:
                    param_frequency[param] = {}
                if value not in param_frequency[param]:
                    param_frequency[param][value] = 0
                param_frequency[param][value] += 1
        
        print("\nMost Frequent Parameters:")
        for param, values in param_frequency.items():
            most_frequent = max(values.items(), key=lambda x: x[1])
            print(f"  {param}: {most_frequent[0]} ({most_frequent[1]}/{len(self.results)} times)")
        
        # 保存结果
        results_df = pd.DataFrame([
            {
                'split_idx': r['split_idx'],
                'train_start': r['train_start'],
                'train_end': r['train_end'],
                'test_start': r['test_start'],
                'test_end': r['test_end'],
                'test_annualized_return': r['test_result']['annualized_return'],
                'test_win_rate': r['test_result']['win_rate'],
                'test_profit_factor': r['test_result']['profit_factor'],
                'test_max_drawdown': r['test_result']['max_drawdown'],
                'test_trades': r['test_result']['total_trades'],
                'best_score': r['best_score']
            }
            for r in self.results
        ])
        
        results_df.to_csv('walk_forward_results.csv', index=False)
        print(f"\nResults saved to walk_forward_results.csv")
        
        return results_df

def generate_signals(df, params):
    """生成信号"""
    # 4H主信号
    cond_4h = (
        (df['obv'] > df['obv'].rolling(14).mean()) &
        (df['ema20'] > df['ema60']) &
        (df['atr'] > params['atr_threshold'] * df['close']) &
        (df['bb'] > params['bb_threshold'])
    )
    
    # 1H funding
    cond_1h = (df['funding'] < params['funding_threshold'])
    
    # 15m突破+量
    df['volume_4h_ma'] = df['volume'].rolling(20).mean()
    cond_15m = df['breakout_15m'] & (df['volmean_15m'] > params['volume_threshold'] * df['volume_4h_ma'])
    
    # 联合信号
    df['signal'] = cond_4h & cond_1h & cond_15m
    
    return df

def backtest(df, params):
    """回测"""
    capital = INIT_CAP
    trades = []
    position = 0.0
    entry_price = 0.0
    entry_time = None
    stop = tp = trail = 0.0
    
    for i in range(30, len(df)):
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']
        
        # 平仓逻辑
        if position > 0:
            hit_tp = price >= tp
            hit_sl = price <= stop
            hit_trail = price <= trail
            exit_flag = hit_tp or hit_sl or hit_trail
            if exit_flag:
                pnl = (price - entry_price) * position
                fee = FEE_RATE * position * (entry_price + price)
                capital += pnl - fee
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'size': position,
                    'pnl': pnl - fee,
                    'reason': 'tp' if hit_tp else ('sl' if hit_sl else 'trail')
                })
                position = 0.0
        
        # 开仓逻辑
        if position == 0 and row['signal']:
            pos_size = capital * 0.015 / (atr * params['sl_mult'] * price) if atr > 0 else 0
            pos_size = min(pos_size, capital / price)
            
            if pos_size * price < MIN_POSITION:
                pos_size = MIN_POSITION / price
            if pos_size * price > MAX_POSITION:
                pos_size = MAX_POSITION / price
                
            position = pos_size
            entry_price = price
            entry_time = row['timestamp']
            stop = price - params['sl_mult'] * atr
            tp = price + params['tp_mult'] * atr
            trail = price - params['tp_mult'] * atr
    
    # 收盘强平
    if position > 0:
        price = df.iloc[-1]['close']
        pnl = (price - entry_price) * position
        fee = FEE_RATE * position * (entry_price + price)
        capital += pnl - fee
        trades.append({
            'entry_time': entry_time,
            'exit_time': df.iloc[-1]['timestamp'],
            'entry_price': entry_price,
            'exit_price': price,
            'size': position,
            'pnl': pnl - fee,
            'reason': 'close'
        })
    
    return trades, capital

def calculate_metrics(trades, final_cap):
    """计算指标"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'sharpe_ratio': 0
        }
    
    total_pnl = sum(t['pnl'] for t in trades)
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades)
    
    # Profit Factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max Drawdown
    capital = INIT_CAP
    peak = capital
    max_dd = 0
    for trade in trades:
        capital += trade['pnl']
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Total Return
    total_return = (final_cap - INIT_CAP) / INIT_CAP
    
    # Sharpe Ratio (简化版)
    returns = [t['pnl'] / INIT_CAP for t in trades]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'total_return': total_return,
        'sharpe_ratio': sharpe
    }

def walk_forward_validation(df, train_days=180, test_days=60):
    """Walk-Forward验证"""
    results = []
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    current_date = start_date + timedelta(days=train_days)
    
    while current_date + timedelta(days=test_days) <= end_date:
        # 训练集
        train_end = current_date
        train_start = train_end - timedelta(days=train_days)
        train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] < train_end)].copy()
        
        # 测试集
        test_start = current_date
        test_end = test_start + timedelta(days=test_days)
        test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)].copy()
        
        if len(train_df) > 30 and len(test_df) > 10:
            # 网格搜索最佳参数
            best_params = None
            best_score = -float('inf')
            
            for params in itertools.product(*PARAM_GRID.values()):
                param_dict = dict(zip(PARAM_GRID.keys(), params))
                
                # 训练集回测
                train_df_signal = generate_signals(train_df, param_dict)
                train_trades, train_cap = backtest(train_df_signal, param_dict)
                train_metrics = calculate_metrics(train_trades, train_cap)
                
                # 评分函数
                score = train_metrics['profit_factor'] * train_metrics['win_rate'] * (1 - train_metrics['max_drawdown'])
                
                if score > best_score:
                    best_score = score
                    best_params = param_dict
            
            # 测试集回测
            if best_params:
                test_df_signal = generate_signals(test_df, best_params)
                test_trades, test_cap = backtest(test_df_signal, best_params)
                test_metrics = calculate_metrics(test_trades, test_cap)
                
                results.append({
                    'period_start': test_start,
                    'period_end': test_end,
                    'params': best_params,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                })
        
        current_date += timedelta(days=test_days)
    
    return results

def monte_carlo_simulation(trades, n_simulations=1000):
    """蒙特卡洛模拟"""
    if not trades:
        return []
    
    results = []
    trade_pnls = [t['pnl'] for t in trades]
    
    for _ in range(n_simulations):
        # 随机打乱交易顺序
        shuffled_pnls = random.sample(trade_pnls, len(trade_pnls))
        
        capital = INIT_CAP
        peak = capital
        max_dd = 0
        
        for pnl in shuffled_pnls:
            capital += pnl
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            if dd > max_dd:
                max_dd = dd
        
        results.append({
            'final_capital': capital,
            'total_return': (capital - INIT_CAP) / INIT_CAP,
            'max_drawdown': max_dd
        })
    
    return results

def main():
    print("开始Walk-Forward网格搜索和蒙特卡洛回测...")
    
    # 加载数据
    df = pd.read_csv('/Users/qiutianyu/ETHUSDT-4h/merged_4h_2023_2025.csv', parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in ['open','high','low','close','volume','obv','ema20','ema60','atr','bb','funding','high_15m','volmean_15m','breakout_15m','volume_surge_15m']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"数据范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # Walk-Forward验证
    print("\n开始Walk-Forward验证...")
    wf_results = walk_forward_validation(df)
    
    if wf_results:
        # 统计Walk-Forward结果
        test_metrics = [r['test_metrics'] for r in wf_results]
        avg_win_rate = np.mean([m['win_rate'] for m in test_metrics])
        avg_profit_factor = np.mean([m['profit_factor'] for m in test_metrics if m['profit_factor'] != float('inf')])
        avg_max_dd = np.mean([m['max_drawdown'] for m in test_metrics])
        
        print(f"Walk-Forward结果:")
        print(f"平均胜率: {avg_win_rate:.2%}")
        print(f"平均Profit Factor: {avg_profit_factor:.2f}")
        print(f"平均最大回撤: {avg_max_dd:.2%}")
        
        # 保存结果
        wf_df = pd.DataFrame(wf_results)
        wf_df.to_csv('walk_forward_results.csv', index=False)
        print("Walk-Forward结果已保存到: walk_forward_results.csv")
    
    # 使用最佳参数进行全样本回测
    print("\n使用最佳参数进行全样本回测...")
    best_params = {
        'atr_threshold': 0.005,
        'bb_threshold': 3,
        'funding_threshold': 0.00005,
        'volume_threshold': 0.15,
        'sl_mult': 0.5,
        'tp_mult': 5.0
    }
    
    df_signal = generate_signals(df, best_params)
    trades, final_cap = backtest(df_signal, best_params)
    metrics = calculate_metrics(trades, final_cap)
    
    print(f"全样本回测结果:")
    print(f"总交易: {metrics['total_trades']}")
    print(f"胜率: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"总收益: {metrics['total_return']:.2%}")
    
    # 蒙特卡洛模拟
    print("\n开始蒙特卡洛模拟...")
    mc_results = monte_carlo_simulation(trades, 1000)
    
    if mc_results:
        mc_df = pd.DataFrame(mc_results)
        print(f"蒙特卡洛模拟结果 (1000次):")
        print(f"平均总收益: {mc_df['total_return'].mean():.2%}")
        print(f"收益标准差: {mc_df['total_return'].std():.2%}")
        print(f"平均最大回撤: {mc_df['max_drawdown'].mean():.2%}")
        print(f"Profit Factor > 1.3的比例: {(mc_df['total_return'] > 0.3).mean():.2%}")
        print(f"最大回撤 < 10%的比例: {(mc_df['max_drawdown'] < 0.1).mean():.2%}")
        
        # 保存蒙特卡洛结果
        mc_df.to_csv('monte_carlo_results.csv', index=False)
        print("蒙特卡洛结果已保存到: monte_carlo_results.csv")

if __name__ == '__main__':
    main()
