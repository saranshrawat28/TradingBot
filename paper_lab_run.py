"""
ApexTrade Paper Trading Accuracy Lab — Master CLI Entrypoint
Usage:
    python paper_lab_run.py                  # Start continuous background scheduler
    python paper_lab_run.py --pick-now       # Generate picks & market open fills immediately
    python paper_lab_run.py --evaluate-now   # Run chronological 1m/5m candle replay evaluation
    python paper_lab_run.py --track-now      # Capture live snapshot of active picks
    python paper_lab_run.py --report-now     # Generate weekly diagnostic accuracy report
    python paper_lab_run.py --days 14        # Custom lookback window for report (e.g. 14 or 28 days)
    python paper_lab_run.py --status         # View today's picks & DB status
"""

import sys
import argparse
from datetime import datetime

from src.paper_lab.daily_picker import DailyPicker
from src.paper_lab.chronological_evaluator import ChronologicalEvaluator
from src.paper_lab.live_tracker import LiveTracker
from src.paper_lab.report_generator import ReportGenerator
from src.paper_lab.paper_db import PaperDB
from src.paper_lab.scheduler import run_scheduler_loop
from src.paper_lab.lab_config import LabConfig
from src.utils.helpers import get_ist_now, format_currency_inr

def safe_print(text: str = ""):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

def print_status():
    now = get_ist_now()
    today_str = now.strftime("%Y-%m-%d")
    picks = PaperDB.get_picks_by_date(today_str)
    all_dates = PaperDB.get_all_dates()

    safe_print("=" * 70)
    safe_print(f"APEXTRADE PAPER LAB STATUS -- {today_str} ({now.strftime('%H:%M:%S')} IST)")
    safe_print(f"* Active Config: {LabConfig.CONFIG_VERSION} | Total Tracked Days in DB: {len(all_dates)}")
    safe_print("=" * 70)

    if not picks:
        safe_print("No picks recorded for today yet. Use `python paper_lab_run.py --pick-now` to generate.")
    else:
        safe_print(f"Today's {len(picks)} Paper Recommendations (Allocated: Rs {len(picks)*20000:,.2f}):\n")
        for i, p in enumerate(picks, 1):
            ep = p.get('entry_price')
            ep_str = f"Rs {ep:,.2f}" if ep else f"Rs {p['signal_price']:,.2f} (Signal)"
            safe_print(f"  {i}. {p['symbol']} ({p['display_name']}) | Score: {p['advisor_score']:.1f}/10 [{p['status']}]")
            safe_print(f"     Entry: {ep_str} | T1: Rs {p['target_1']:,.2f} | T2: Rs {p['target_2']:,.2f} | SL: Rs {p['stop_loss']:,.2f} | Qty: {p['quantity']}")
            if p.get("top_signals"):
                safe_print(f"     Key Catalysts: {', '.join(p['top_signals'][:2])}")
            safe_print()

def main():
    parser = argparse.ArgumentParser(description="ApexTrade Paper Trading Accuracy Lab CLI")
    parser.add_argument("--pick-now", action="store_true", help="Generate Top 5 picks and confirm open fills immediately")
    parser.add_argument("--force", action="store_true", help="Force pick generation even on weekends/holidays (for manual testing)")
    parser.add_argument("--evaluate-now", action="store_true", help="Run chronological candle replay outcome resolution")
    parser.add_argument("--track-now", action="store_true", help="Capture live price snapshot of active picks")
    parser.add_argument("--report-now", action="store_true", help="Generate self-diagnostic accuracy report")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days for report generation (default: 7)")
    parser.add_argument("--status", action="store_true", help="Display current status and today's picks")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.pick_now:
        print("[PaperLab CLI] Executing on-demand daily picker & open fill...")
        DailyPicker.run_daily_picker_catchup(force=args.force)
        print_status()
        return

    if args.evaluate_now:
        print("[PaperLab CLI] Executing chronological candle outcome replay...")
        ChronologicalEvaluator.evaluate_all_picks_for_date()
        return

    if args.track_now:
        print("[PaperLab CLI] Tracking live snapshots...")
        LiveTracker.track_active_picks()
        print_status()
        return

    if args.report_now or args.days != 7:
        print(f"[PaperLab CLI] Generating accuracy & diagnostic report for past {args.days} days...")
        rep = ReportGenerator.generate_report(days_lookback=args.days)
        safe_print("\n" + rep.get("markdown_text", ""))
        return

    # Default: Run full background scheduler loop
    run_scheduler_loop()

if __name__ == "__main__":
    main()
