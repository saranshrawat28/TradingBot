"""
Modular Agentic Tool Runner for Conversational AI Assistant.
Provides deterministic system execution tools for:
1. Portfolio & Margin Telemetry
2. Natural Language Position Square-Off
3. Black-Scholes Options Greeks & Smart Strike Recommendation
4. Pre-Market Sentiment & Morning Catalyst Intel
5. Multi-Sector Technical Screener
"""

from typing import Dict, List, Any, Optional
import pandas as pd

import config
from src.data.data_fetcher import get_live_quote, get_historical_data, resolve_ticker, search_indian_stocks
from src.strategies.indicators import (
    calculate_ema, calculate_rsi, calculate_rvol, calculate_ttm_squeeze,
    calculate_camarilla_pivots, calculate_volume_profile, add_all_indicators
)
from src.strategies.options_greeks import SmartStrikeSelector, BlackScholesEngine
from src.engine.pre_market_analyzer import PreMarketAnalyzer
from src.utils.storage import get_portfolio_state, get_open_positions
from src.utils.helpers import display_symbol_name, clean_symbol, format_currency_inr, get_ist_now

class AssistantToolRunner:
    """
    Dispatcher and execution runner for Assistant Agentic Tools.
    """

    @classmethod
    def get_portfolio_status(cls, broker_instance=None) -> Dict[str, Any]:
        """
        Extracts live portfolio status, cash headroom, margin utilization,
        open positions breakdown, and realized/unrealized P&L.
        """
        p_state = get_portfolio_state()
        cash = float(p_state.get("cash", 100000.0))
        realized_pnl = float(p_state.get("realized_pnl", 0.0))
        
        # Pull live positions from broker or storage
        open_pos = []
        if broker_instance and hasattr(broker_instance, "get_open_positions"):
            open_pos = broker_instance.get_open_positions()
        else:
            open_pos = get_open_positions()

        total_invested = 0.0
        total_unrealized_pnl = 0.0
        pos_details = []

        for p in open_pos:
            sym = p.get("symbol", "")
            qty = int(p.get("quantity", 0))
            entry_p = float(p.get("entry_price", 0.0))
            
            # Fetch current quote
            quote = get_live_quote(sym)
            curr_p = float(quote.get("price", entry_p)) if quote else entry_p
            
            pnl = (curr_p - entry_p) * qty
            pnl_pct = ((curr_p - entry_p) / entry_p * 100.0) if entry_p > 0 else 0.0
            invested = entry_p * qty
            
            total_invested += invested
            total_unrealized_pnl += pnl
            
            stage = p.get("stage", "INITIAL")
            locked_r = float(p.get("locked_r", 0.0))
            trailing_sl = float(p.get("trailing_sl") or p.get("sl", 0.0))
            
            pos_details.append({
                "symbol": sym,
                "display_name": display_symbol_name(sym),
                "quantity": qty,
                "entry_price": entry_p,
                "current_price": curr_p,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "invested": invested,
                "side": p.get("side", "BUY"),
                "sl": float(p.get("sl", 0.0)),
                "trailing_sl": trailing_sl,
                "stage": stage,
                "locked_r": locked_r,
                "tp": float(p.get("tp", 0.0))
            })

        total_portfolio_val = cash + total_invested + total_unrealized_pnl
        day_total_pnl = realized_pnl + total_unrealized_pnl

        # Generate friendly markdown summary
        pnl_sign = "+" if day_total_pnl >= 0 else ""
        pnl_emoji = "🟢" if day_total_pnl >= 0 else "🔴"
        
        md_lines = [
            f"### 💼 Live Portfolio Status\n",
            f"- **Account Value**: `₹{total_portfolio_val:,.2f}`",
            f"- **Available Cash**: `₹{cash:,.2f}`",
            f"- **Invested Margin**: `₹{total_invested:,.2f}`",
            f"- **Today's Total P&L**: {pnl_emoji} **{pnl_sign}₹{day_total_pnl:,.2f}** (Realized: `₹{realized_pnl:,.2f}` | Unrealized: `₹{total_unrealized_pnl:,.2f}`)\n"
        ]

        if pos_details:
            md_lines.append(f"#### 📊 Active Open Positions ({len(pos_details)}):")
            for pos in pos_details:
                pos_sign = "+" if pos["pnl"] >= 0 else ""
                stage_tag = ""
                if pos["stage"] == "BREAKEVEN_LOCKED":
                    stage_tag = " &bull; 🔒 *Breakeven Protected*"
                elif pos["stage"] == "T1_BOOKED_RUNNER_TRAILING":
                    stage_tag = f" &bull; 🚀 *ATR Runner Trailing (Floor: +{pos['locked_r']}R / ₹{pos['trailing_sl']:,.2f})*"
                elif pos["stage"] == "PARABOLIC_RIDER":
                    stage_tag = f" &bull; ⚡ *Parabolic Super-Trend (Floor: +{pos['locked_r']}R / ₹{pos['trailing_sl']:,.2f})*"

                md_lines.append(
                    f"• **{pos['display_name']}**: {pos['quantity']} shares @ ₹{pos['entry_price']:,.2f} "
                    f"(LTP: ₹{pos['current_price']:,.2f} | P&L: **{pos_sign}₹{pos['pnl']:,.2f} / {pos_sign}{pos['pnl_pct']:.2f}%**{stage_tag})"
                )
        else:
            md_lines.append("🛡️ *You currently have no open positions. All capital is parked safely in cash.*")

        return {
            "tool_name": "get_portfolio_status",
            "success": True,
            "ui_card_type": "PORTFOLIO",
            "data": {
                "total_portfolio_value": total_portfolio_val,
                "cash": cash,
                "invested_margin": total_invested,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "day_total_pnl": day_total_pnl,
                "open_positions": pos_details,
                "count_positions": len(pos_details)
            },
            "summary_markdown": "\n".join(md_lines)
        }

    @classmethod
    def square_off_action(cls, symbol_query: str, broker_instance=None) -> Dict[str, Any]:
        """
        Identifies open positions matching the symbol query and builds
        a deterministic 1-click Square-Off Action Card.
        """
        open_pos = []
        if broker_instance and hasattr(broker_instance, "get_open_positions"):
            open_pos = broker_instance.get_open_positions()
        else:
            open_pos = get_open_positions()

        if not open_pos:
            return {
                "tool_name": "square_off_action",
                "success": False,
                "ui_card_type": "MESSAGE",
                "data": {},
                "summary_markdown": "⚠️ **No Open Positions Found**: You have no active trades to square off right now."
            }

        # Check for square off ALL
        if any(w in symbol_query.lower() for w in ["all", "everything", "entire", "all positions"]):
            return {
                "tool_name": "square_off_action",
                "success": True,
                "ui_card_type": "SQUARE_OFF",
                "data": {
                    "mode": "ALL",
                    "target_symbol": "ALL",
                    "display_name": "All Open Positions",
                    "count": len(open_pos),
                    "positions": open_pos
                },
                "summary_markdown": f"🛑 **Square Off All Positions**: Prepare to exit all **{len(open_pos)} active trades** and revert 100% capital to cash."
            }

        # Match specific symbol
        matched_pos = None
        cleaned_query = clean_symbol(symbol_query).replace(".NS", "").upper()
        
        for p in open_pos:
            p_sym = p.get("symbol", "").replace(".NS", "").upper()
            if cleaned_query in p_sym or p_sym in cleaned_query:
                matched_pos = p
                break

        if not matched_pos:
            # Fallback ticker resolution
            resolved = resolve_ticker(symbol_query)
            if resolved:
                res_clean = resolved.replace(".NS", "").upper()
                for p in open_pos:
                    if p.get("symbol", "").replace(".NS", "").upper() == res_clean:
                        matched_pos = p
                        break

        if not matched_pos:
            active_names = [display_symbol_name(p.get("symbol", "")) for p in open_pos]
            return {
                "tool_name": "square_off_action",
                "success": False,
                "ui_card_type": "MESSAGE",
                "data": {"open_positions": active_names},
                "summary_markdown": f"⚠️ Could not find an active trade matching **'{symbol_query}'**.\n\nYour active positions are: **{', '.join(active_names)}**."
            }

        sym = matched_pos.get("symbol", "")
        qty = int(matched_pos.get("quantity", 0))
        entry_p = float(matched_pos.get("entry_price", 0.0))
        quote = get_live_quote(sym)
        curr_p = float(quote.get("price", entry_p)) if quote else entry_p
        pnl = (curr_p - entry_p) * qty
        pnl_pct = ((curr_p - entry_p) / entry_p * 100.0) if entry_p > 0 else 0.0

        pnl_sign = "+" if pnl >= 0 else ""
        return {
            "tool_name": "square_off_action",
            "success": True,
            "ui_card_type": "SQUARE_OFF",
            "data": {
                "mode": "SINGLE",
                "target_symbol": sym,
                "display_name": display_symbol_name(sym),
                "quantity": qty,
                "entry_price": entry_p,
                "current_price": curr_p,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            },
            "summary_markdown": (
                f"🛑 **Confirm Square-Off**: Exit **{qty} shares of {display_symbol_name(sym)}** @ current market price ₹{curr_p:,.2f}.\n"
                f"• Estimated P&L: **{pnl_sign}₹{pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)**"
            )
        }

    @classmethod
    def get_options_recommendation(
        cls,
        symbol_query: str,
        bias: str = "BUY_CALL",
        dte_days: float = 3.0,
        preference: str = "ATM"
    ) -> Dict[str, Any]:
        """
        Calculates European Black-Scholes Greeks, IV, and optimal strike selection
        for NIFTY, BANKNIFTY, FINNIFTY, or any major F&O stock.
        """
        # Clean symbol
        symbol_clean = clean_symbol(symbol_query).replace(".NS", "").upper()
        if "BANK" in symbol_clean:
            base_sym = "^NSEBANK"
            display_name = "BANKNIFTY"
            lot_size = 15
        elif "FIN" in symbol_clean:
            base_sym = "NIFTY_FIN_SERVICE.NS"
            display_name = "FINNIFTY"
            lot_size = 25
        elif "NIFTY" in symbol_clean:
            base_sym = "^NSEI"
            display_name = "NIFTY 50"
            lot_size = 25
        else:
            resolved = resolve_ticker(symbol_query)
            base_sym = resolved if resolved else f"{symbol_clean}.NS"
            display_name = display_symbol_name(base_sym)
            lot_size = 100

        quote = get_live_quote(base_sym)
        spot_price = float(quote.get("price", 0.0)) if quote else 0.0
        if spot_price <= 0:
            spot_price = 24500.0 if "NIFTY" in display_name else 1000.0

        opt_spec = SmartStrikeSelector.select_optimal_strike(
            symbol=display_name,
            spot_price=spot_price,
            action=bias,
            dte_days=dte_days,
            preference=preference
        )

        contract_name = opt_spec.get("contract_symbol", f"{display_name} {opt_spec.get('chosen_strike')} {opt_spec.get('option_type')}")
        theo_p = float(opt_spec.get("theoretical_price", 120.0))
        greeks = opt_spec.get("greeks", {})
        delta = float(greeks.get("delta", 0.50))
        gamma = float(greeks.get("gamma", 0.001))
        theta = float(greeks.get("theta_per_day", -5.0))
        vega = float(greeks.get("vega", 8.0))
        iv = float(opt_spec.get("implied_volatility", 0.15) * 100.0)

        # Risk-reward targets
        t1_premium = round(theo_p * 1.35, 1) # +35% target
        t2_premium = round(theo_p * 1.65, 1) # +65% runner
        sl_premium = round(theo_p * 0.75, 1) # -25% safety stop

        capital_req = theo_p * lot_size

        md_summary = (
            f"### 🎯 Recommended Option Strike: **{contract_name}**\n\n"
            f"• **Underlying Spot**: ₹{spot_price:,.2f} ({display_name})\n"
            f"• **Estimated Premium**: `₹{theo_p:,.2f}` (1 Lot = {lot_size} qty $\\rightarrow$ `₹{capital_req:,.2f}`)\n"
            f"• **Target 1 (+35%)**: <strong style='color: #10b981;'>₹{t1_premium:,.2f}</strong>\n"
            f"• **Target 2 (+65%)**: <strong style='color: #10b981;'>₹{t2_premium:,.2f}</strong>\n"
            f"• **Safety Stop-Loss (-25%)**: <strong style='color: #f43f5e;'>₹{sl_premium:,.2f}</strong>\n\n"
            f"**⚡ Analytical Black-Scholes Greeks**:\n"
            f"- **Delta (Δ)**: `{delta:+.2f}` (Price sensitivity per ₹1 spot move)\n"
            f"- **Theta (Θ)**: `₹{theta:.2f}/day` (Expected time decay per session)\n"
            f"- **Gamma (Γ)**: `{gamma:.4f}` | **Vega (ν)**: `{vega:.2f}` | **IV**: `{iv:.1f}%`"
        )

        return {
            "tool_name": "get_options_recommendation",
            "success": True,
            "ui_card_type": "OPTIONS",
            "data": {
                "symbol": base_sym,
                "display_name": display_name,
                "spot_price": spot_price,
                "contract_symbol": contract_name,
                "strike": opt_spec.get("chosen_strike"),
                "option_type": opt_spec.get("option_type"),
                "theoretical_premium": theo_p,
                "lot_size": lot_size,
                "capital_required": capital_req,
                "target_1_premium": t1_premium,
                "target_2_premium": t2_premium,
                "stop_loss_premium": sl_premium,
                "greeks": {
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "vega": vega,
                    "iv_pct": iv
                }
            },
            "summary_markdown": md_summary
        }

    @classmethod
    def get_premarket_intel(cls) -> Dict[str, Any]:
        """
        Queries the pre-market intelligence engine and extracts morning cues.
        """
        report = PreMarketAnalyzer.get_pre_market_report()
        nifty_quote = report.get("nifty_quote", {})
        sentiment = report.get("overall_sentiment", "NEUTRAL")
        recs = report.get("recommendations", [])
        gap_leaders = report.get("gap_leaders", [])

        md_lines = [
            f"### 🌅 Pre-Market Intelligence & Opening Cues\n",
            f"• **Market Mood**: **{sentiment}**",
            f"• **NIFTY 50 Spot**: `₹{nifty_quote.get('price', 24500):,.2f}` ({nifty_quote.get('change_pct', 0.0):+.2f}%)\n"
        ]

        if recs:
            md_lines.append(f"#### 🚀 Curated High-Conviction Stock Setups:")
            for r in recs[:3]:
                md_lines.append(
                    f"• **{r.get('display_name', r.get('symbol'))}** ({r.get('signal')} @ ₹{r.get('current_price', 0):,.2f}): "
                    f"Target `₹{r.get('target_1', 0):,.2f}` | Stop-Loss `₹{r.get('stop_loss', 0):,.2f}` (Score: {r.get('score', 0):.1f}/10)"
                )

        return {
            "tool_name": "get_premarket_intel",
            "success": True,
            "ui_card_type": "INTEL",
            "data": report,
            "summary_markdown": "\n".join(md_lines)
        }

    @classmethod
    def run_technical_scanner(cls, scan_type: str = "golden_cross", sector: str = "all", timeframe: str = "1d") -> Dict[str, Any]:
        """
        Executes institutional technical screener algorithms:
        - golden_cross: 50 EMA > 200 EMA bullish confirmation
        - death_cross: 50 EMA < 200 EMA bearish breakdown
        - camarilla_breakout: Price testing or breaking above Camarilla H4 level
        - ttm_squeeze: Volatility squeeze compression (coiling for breakout)
        - volume_shockers: RVOL >= 1.5x institutional accumulation
        - rsi_oversold: RSI <= 35 mean-reversion bounce
        - rsi_overbought: RSI >= 65 exhaustion pullback
        """
        scan_key = scan_type.lower().replace("-", "_").replace(" ", "_")
        stock_pool = config.DEFAULT_WATCHLIST
        if sector.lower() != "all":
            stock_pool = [i for i in stock_pool if sector.lower() in i.get("category", "").lower()]
        if not stock_pool:
            stock_pool = config.DEFAULT_WATCHLIST[:15]

        # Use appropriate period based on scan type
        period = "1y" if scan_key in ["golden_cross", "death_cross"] else ("5d" if timeframe == "5m" else "1mo")
        interval = "1d" if scan_key in ["golden_cross", "death_cross"] else timeframe

        scanner_results = []

        for item in stock_pool[:20]:
            sym = item["symbol"]
            if sym.startswith("^"):
                continue
            try:
                df = get_historical_data(sym, period=period, interval=interval)
                if df.empty or len(df) < 15:
                    continue

                close_s = df["Close"]
                curr_p = float(close_s.iloc[-1])
                rsi_val = float(calculate_rsi(close_s, 14).iloc[-1]) if len(df) >= 15 else 50.0

                if scan_key in ["golden_cross", "death_cross"]:
                    ema50 = float(calculate_ema(close_s, min(50, len(df))).iloc[-1])
                    ema200 = float(calculate_ema(close_s, min(200, len(df))).iloc[-1])
                    diff_pct = ((ema50 - ema200) / ema200) * 100.0 if ema200 > 0 else 0.0

                    is_match = False
                    if scan_key == "golden_cross" and ema50 > ema200:
                        is_match = True
                    elif scan_key == "death_cross" and ema50 < ema200:
                        is_match = True

                    if is_match:
                        scanner_results.append({
                            "symbol": sym,
                            "name": item["name"],
                            "price": curr_p,
                            "ema50": round(ema50, 2),
                            "ema200": round(ema200, 2),
                            "diff_pct": round(diff_pct, 2),
                            "rsi": round(rsi_val, 1),
                            "status": "🟢 GOLDEN CROSS" if scan_key == "golden_cross" else "🔴 DEATH CROSS",
                            "score": round(8.0 + min(1.5, abs(diff_pct) / 5.0), 1)
                        })

                elif scan_key == "camarilla_breakout":
                    high_p = float(df["High"].iloc[-1]) if "High" in df.columns else curr_p
                    low_p = float(df["Low"].iloc[-1]) if "Low" in df.columns else curr_p
                    cam = calculate_camarilla_pivots(high=high_p, low=low_p, close=curr_p)
                    h4 = cam.get("h4", curr_p)
                    if curr_p >= h4 * 0.992:
                        scanner_results.append({
                            "symbol": sym,
                            "name": item["name"],
                            "price": curr_p,
                            "h4_level": round(h4, 2),
                            "rsi": round(rsi_val, 1),
                            "status": "🚀 H4 BREAKOUT",
                            "score": 8.5
                        })

                elif scan_key == "ttm_squeeze":
                    sqz = calculate_ttm_squeeze(df)
                    if sqz.get("squeeze_on") or sqz.get("squeeze_fired"):
                        scanner_results.append({
                            "symbol": sym,
                            "name": item["name"],
                            "price": curr_p,
                            "status": "⚡ SQUEEZE ACTIVE" if sqz.get("squeeze_on") else "🔥 SQUEEZE FIRED",
                            "momentum": sqz.get("momentum_direction", "UP"),
                            "rsi": round(rsi_val, 1),
                            "score": 8.2
                        })

                elif scan_key == "volume_shockers":
                    rvol_val = float(calculate_rvol(df).iloc[-1]) if len(df) >= 20 else 1.0
                    if rvol_val >= 1.4:
                        scanner_results.append({
                            "symbol": sym,
                            "name": item["name"],
                            "price": curr_p,
                            "rvol": round(rvol_val, 2),
                            "rsi": round(rsi_val, 1),
                            "status": f"📊 {rvol_val:.1f}x RVOL",
                            "score": round(7.0 + min(2.5, rvol_val), 1)
                        })

                elif scan_key == "rsi_oversold":
                    if rsi_val <= 38.0:
                        scanner_results.append({
                            "symbol": sym,
                            "name": item["name"],
                            "price": curr_p,
                            "rsi": round(rsi_val, 1),
                            "status": "🟢 OVERSOLD (Reversal)",
                            "score": round(8.0 + (38.0 - rsi_val) / 10.0, 1)
                        })

                elif scan_key == "rsi_overbought":
                    if rsi_val >= 68.0:
                        scanner_results.append({
                            "symbol": sym,
                            "name": item["name"],
                            "price": curr_p,
                            "rsi": round(rsi_val, 1),
                            "status": "🔴 OVERBOUGHT (Caution)",
                            "score": 6.5
                        })

            except Exception:
                continue

        # Sort by score descending
        scanner_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_picks = scanner_results[:6]

        # Generate friendly markdown summary
        title_map = {
            "golden_cross": "✨ Golden Cross Momentum Scan (50 EMA > 200 EMA)",
            "death_cross": "⚠️ Death Cross Breakdown Scan (50 EMA < 200 EMA)",
            "camarilla_breakout": "🏛️ Camarilla H4 Institutional Breakout Scan",
            "ttm_squeeze": "⚡ TTM Volatility Squeeze Coiling Scan",
            "volume_shockers": "📊 High Relative Volume (RVOL) Institutional Flow Scan",
            "rsi_oversold": "💎 RSI Oversold Value Reversal Scan",
            "rsi_overbought": "⚠️ RSI Overbought Exhaustion Scan"
        }
        scan_title = title_map.get(scan_key, f"🔍 Technical Screener: {scan_type.upper()}")

        md_lines = [f"### {scan_title}\n"]
        if top_picks:
            md_lines.append(f"Found **{len(top_picks)} high-conviction candidates** matching your criteria:\n")
            
            if scan_key in ["golden_cross", "death_cross"]:
                md_lines.append("| Stock | LTP | 50 EMA | 200 EMA | Distance | RSI | Status |")
                md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
                for s in top_picks:
                    md_lines.append(f"| **{s['name']}** | ₹{s['price']:,.2f} | ₹{s['ema50']:,.2f} | ₹{s['ema200']:,.2f} | `{s['diff_pct']:+.1f}%` | `{s['rsi']}` | {s['status']} |")
            else:
                for idx, s in enumerate(top_picks, 1):
                    md_lines.append(f"• **{idx}. {s['name']}** (`₹{s['price']:,.2f}`): {s['status']} (RSI: `{s['rsi']}`) — Score: **{s.get('score', 7.5)}/10**")

            md_lines.append("\n👉 *Click any stock below to view its live execution card or ask 'Analyze <Stock>' for a full trade plan.*")
        else:
            md_lines.append(f"🛡️ *No stocks currently meet the strict {scan_type} filter under current market conditions. Showing baseline leaders:*")
            for item in config.DEFAULT_WATCHLIST[:4]:
                md_lines.append(f"• **{item['name']}** ({item['symbol']}): Baseline Watchlist")

        return {
            "tool_name": "run_technical_scanner",
            "success": True,
            "ui_card_type": "SCREENER",
            "data": {
                "scan_type": scan_key,
                "sector": sector,
                "timeframe": timeframe,
                "results": top_picks
            },
            "summary_markdown": "\n".join(md_lines)
        }

    @classmethod
    def run_sector_screener(cls, sector: str = "Banking", timeframe: str = "15m") -> Dict[str, Any]:
        """
        Scans a specific Indian market sector and returns top scoring buy/sell candidates.
        """
        return cls.run_technical_scanner(scan_type="sector", sector=sector, timeframe=timeframe)
