"""
High-Performance Backtesting and Quantitative Performance Analytics Engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import config
from src.strategies.base_strategy import BaseStrategy
from src.engine.risk_manager import RiskManager

class Backtester:
    """
    Backtesting engine simulating bar-by-bar execution with SL/TP, trailing stops,
    and Indian regulatory tax & slippage deductions.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = config.DEFAULT_INITIAL_CAPITAL,
        risk_manager: RiskManager = None,
        enable_shorting: bool = False
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_manager = risk_manager or RiskManager()
        self.enable_shorting = enable_shorting

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run backtest on OHLCV DataFrame.
        Returns comprehensive performance metrics, trade journal, and equity curve.
        """
        if df.empty or len(df) < 20:
            return self._empty_result()
            
        # Generate Signals
        data = self.strategy.generate_signals(df)
        if "Signal" not in data.columns:
            return self._empty_result()
            
        cash = self.initial_capital
        position = None  # None or dict
        trades = []
        equity_series = []
        
        benchmark_initial_price = float(data["Close"].iloc[0])
        benchmark_shares = self.initial_capital / benchmark_initial_price
        
        fees = config.INDIAN_FEES
        
        for i in range(len(data)):
            current_bar = data.iloc[i]
            date = data.index[i]
            open_p = float(current_bar["Open"])
            high_p = float(current_bar["High"])
            low_p = float(current_bar["Low"])
            close_p = float(current_bar["Close"])
            signal = int(current_bar.get("Signal", 0))
            atr = float(current_bar.get("ATR_14", close_p * 0.015))
            
            # 1. Manage Active Position (Check SL, TP, Trailing Stop)
            if position:
                side = position["side"]
                entry_price = position["entry_price"]
                sl = position["sl"]
                tp = position["tp"]
                qty = position["quantity"]
                highest = max(position["highest_price"], high_p)
                position["highest_price"] = highest
                
                # Update Trailing Stop
                updated_sl = self.risk_manager.update_trailing_stop(
                    side, entry_price, close_p, highest, sl
                )
                position["sl"] = updated_sl
                
                exit_price = None
                exit_reason = ""
                
                if side == "LONG":
                    # Check Stop Loss hit
                    if low_p <= updated_sl:
                        exit_price = updated_sl
                        exit_reason = "Stop-Loss Hit"
                    # Check Take Profit hit
                    elif high_p >= tp:
                        exit_price = tp
                        exit_reason = "Take-Profit Target Hit"
                    # Check Opposite Signal
                    elif signal == -1 or signal == 2:
                        exit_price = close_p
                        exit_reason = "Opposite Strategy Signal"
                elif side == "SHORT":
                    if high_p >= updated_sl:
                        exit_price = updated_sl
                        exit_reason = "Stop-Loss Hit"
                    elif low_p <= tp:
                        exit_price = tp
                        exit_reason = "Take-Profit Target Hit"
                    elif signal == 1 or signal == -2:
                        exit_price = close_p
                        exit_reason = "Opposite Strategy Signal"
                        
                # Execute Exit
                if exit_price is not None:
                    exit_price = round(exit_price, 2)
                    gross_pnl = (exit_price - entry_price) * qty if side == "LONG" else (entry_price - exit_price) * qty
                    
                    # Deduct Indian regulatory taxes & brokerage
                    turnover = exit_price * qty
                    exit_fee = min(fees["brokerage_per_order"], turnover * fees["brokerage_pct"]) + (turnover * fees["stt_intraday_sell_pct"]) + (turnover * fees["exchange_txn_charge_pct"])
                    net_pnl = gross_pnl - exit_fee
                    
                    cash += (entry_price * qty) + net_pnl if side == "LONG" else (position["margin"] + net_pnl)
                    pnl_pct = (net_pnl / (entry_price * qty)) * 100.0
                    
                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": date.strftime("%Y-%m-%d %H:%M") if hasattr(date, "strftime") else str(date),
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": qty,
                        "gross_pnl": round(gross_pnl, 2),
                        "net_pnl": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "exit_reason": exit_reason,
                        "duration_bars": i - position["entry_index"]
                    })
                    position = None
                    
            # 2. Enter New Position
            if position is None:
                if signal == 1:
                    # Long Entry
                    entry_p = close_p * (1.0 + fees["slippage_pct"])
                    sl_p, tp_p = self.risk_manager.calculate_sl_tp_prices("BUY", entry_p, atr)
                    qty = self.risk_manager.calculate_position_size(cash, entry_p, sl_p)
                    
                    order_val = entry_p * qty
                    entry_fee = min(fees["brokerage_per_order"], order_val * fees["brokerage_pct"]) + (order_val * fees["stamp_duty_buy_pct"])
                    
                    if cash >= order_val + entry_fee and qty > 0:
                        cash -= (order_val + entry_fee)
                        position = {
                            "side": "LONG",
                            "entry_date": date.strftime("%Y-%m-%d %H:%M") if hasattr(date, "strftime") else str(date),
                            "entry_price": round(entry_p, 2),
                            "quantity": qty,
                            "sl": sl_p,
                            "tp": tp_p,
                            "highest_price": entry_p,
                            "entry_index": i,
                            "margin": order_val
                        }
                elif signal == -1 and self.enable_shorting:
                    # Short Entry
                    entry_p = close_p * (1.0 - fees["slippage_pct"])
                    sl_p, tp_p = self.risk_manager.calculate_sl_tp_prices("SELL", entry_p, atr)
                    qty = self.risk_manager.calculate_position_size(cash, entry_p, sl_p)
                    
                    margin_req = entry_p * qty * 0.2
                    if cash >= margin_req and qty > 0:
                        cash -= margin_req
                        position = {
                            "side": "SHORT",
                            "entry_date": date.strftime("%Y-%m-%d %H:%M") if hasattr(date, "strftime") else str(date),
                            "entry_price": round(entry_p, 2),
                            "quantity": qty,
                            "sl": sl_p,
                            "tp": tp_p,
                            "highest_price": entry_p,
                            "entry_index": i,
                            "margin": margin_req
                        }
                        
            # 3. Calculate Current Equity Snapshot
            unrealized = 0.0
            if position:
                side = position["side"]
                ep = position["entry_price"]
                q = position["quantity"]
                if side == "LONG":
                    unrealized = (close_p - ep) * q
                    current_equity = cash + (ep * q) + unrealized
                else:
                    unrealized = (ep - close_p) * q
                    current_equity = cash + position["margin"] + unrealized
            else:
                current_equity = cash
                
            benchmark_val = benchmark_shares * close_p
            equity_series.append({
                "Date": date,
                "Equity": round(current_equity, 2),
                "Benchmark_Equity": round(benchmark_val, 2),
                "Close": close_p
            })
            
        # Compile Equity DataFrame
        equity_df = pd.DataFrame(equity_series).set_index("Date")
        
        # Calculate Performance Metrics
        final_equity = equity_df["Equity"].iloc[-1] if not equity_df.empty else self.initial_capital
        net_profit = final_equity - self.initial_capital
        total_return_pct = (net_profit / self.initial_capital) * 100.0
        
        benchmark_final = equity_df["Benchmark_Equity"].iloc[-1] if not equity_df.empty else self.initial_capital
        benchmark_return_pct = ((benchmark_final - self.initial_capital) / self.initial_capital) * 100.0
        
        # Drawdown
        equity_df["Peak"] = equity_df["Equity"].cummax()
        equity_df["Drawdown_Pct"] = ((equity_df["Equity"] - equity_df["Peak"]) / equity_df["Peak"]) * 100.0
        max_drawdown_pct = abs(equity_df["Drawdown_Pct"].min()) if not equity_df.empty else 0.0
        
        # Trade Stats
        total_trades = len(trades)
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate_pct = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0
        
        gross_profit = sum(t["net_pnl"] for t in wins)
        gross_loss = abs(sum(t["net_pnl"] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        
        avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0.0
        risk_reward = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        
        # Daily Returns & Sharpe / Sortino
        daily_returns = equity_df["Equity"].pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
            negative_returns = daily_returns[daily_returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 and negative_returns.std() > 0 else daily_returns.std()
            sortino_ratio = np.sqrt(252) * (daily_returns.mean() / downside_std)
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            
        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "benchmark_return_pct": round(benchmark_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate_pct": round(win_rate_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_trade_pnl": round(net_profit / total_trades, 2) if total_trades > 0 else 0.0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "risk_reward_ratio": round(risk_reward, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "trades": trades,
            "equity_df": equity_df,
            "signals_df": data
        }

    def _empty_result(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.initial_capital,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "risk_reward_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "trades": [],
            "equity_df": pd.DataFrame(),
            "signals_df": pd.DataFrame()
        }
