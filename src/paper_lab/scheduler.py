"""
Master Unattended Background Scheduler for ApexTrade Paper Trading Lab.
Zero external dependency: Built on native Python datetime & time loop.
Manages 08:50 AM signal scan, 09:15 AM fill confirmation, 30m live tracking,
03:35 PM chronological candle outcome evaluation, and Friday weekly reports.
"""

import time
from datetime import datetime
from src.paper_lab.daily_picker import DailyPicker
from src.paper_lab.chronological_evaluator import ChronologicalEvaluator
from src.paper_lab.live_tracker import LiveTracker
from src.paper_lab.report_generator import ReportGenerator
from src.paper_lab.holiday_calendar import is_trading_day
from src.paper_lab.lab_config import LabConfig
from src.utils.helpers import get_ist_now

def job_morning_signals():
    """08:50 AM: Generate Top 5 candidate recommendations."""
    now = get_ist_now()
    is_trade, reason = is_trading_day(now)
    if not is_trade:
        print(f"[Scheduler 08:50 AM] Market holiday / weekend: {reason}. Skipping signal scan.")
        return

    print(f"\n[Scheduler 08:50 AM] Starting Pre-Market Stock Picker...")
    try:
        DailyPicker.generate_morning_signals()
    except Exception as e:
        print(f"[Scheduler 08:50 AM] Error during signal generation: {e}")

def job_market_open_fills():
    """09:15 AM: Confirm real market open fills and anchor levels."""
    now = get_ist_now()
    is_trade, reason = is_trading_day(now)
    if not is_trade:
        return

    print(f"\n[Scheduler 09:15 AM] Confirming 09:15 AM Market Open Fills...")
    try:
        DailyPicker.confirm_market_open_fills()
    except Exception as e:
        print(f"[Scheduler 09:15 AM] Error during fill confirmation: {e}")

def job_live_tracking():
    """Every 30 Mins (09:45 to 15:15): Log intraday snapshot."""
    now = get_ist_now()
    is_trade, _ = is_trading_day(now)
    if not is_trade:
        return

    hm = now.hour * 100 + now.minute
    if 915 <= hm <= 1530:
        try:
            LiveTracker.track_active_picks()
        except Exception as e:
            print(f"[Scheduler Tracker] Error in live tracking: {e}")

def job_eod_chronological_evaluation():
    """03:35 PM: Perform full 1m/5m chronological candle replay outcome resolution."""
    now = get_ist_now()
    is_trade, reason = is_trading_day(now)
    if not is_trade:
        return

    print(f"\n[Scheduler 03:35 PM] Running Chronological Candle Replay Outcome Evaluator...")
    try:
        ChronologicalEvaluator.evaluate_all_picks_for_date()
    except Exception as e:
        print(f"[Scheduler 03:35 PM] Error during EOD chronological evaluation: {e}")

def job_weekly_report():
    """Friday 05:00 PM: Generate weekly accuracy and signal diagnostic report."""
    now = get_ist_now()
    if now.weekday() == 4: # Friday
        print(f"\n[Scheduler 05:00 PM Friday] Generating Weekly Self-Diagnostic Accuracy Report...")
        try:
            ReportGenerator.generate_report(days_lookback=7)
        except Exception as e:
            print(f"[Scheduler Friday] Error generating weekly report: {e}")

def run_scheduler_loop():
    """
    Native loop for background execution (Zero external dependency).
    """
    print("=" * 70)
    print(f"ApexTrade Paper Trading Accuracy Lab Scheduler Started ({LabConfig.CONFIG_VERSION})")
    print(f"* 08:50 AM IST: Pre-Market Signal Generation")
    print(f"* 09:15 AM IST: Market Open Fill Confirmation")
    print(f"* Every 30 Mins: Live Intraday Telemetry Tracking")
    print(f"* 03:35 PM IST: Chronological 1m/5m Candle Replay Resolution")
    print(f"* Friday 05:00 PM: Weekly Accuracy Audit & Diagnostic Report")
    print("=" * 70)

    # Initial startup check & catch-up
    now = get_ist_now()
    is_trade, _ = is_trading_day(now)
    hm = now.hour * 100 + now.minute

    if is_trade and 915 <= hm <= 1530:
        print(f"[Scheduler Startup] Started during live market hours ({now.strftime('%H:%M:%S')}). Running catch-up picker & fill...")
        try:
            DailyPicker.run_daily_picker_catchup()
        except Exception as e:
            print(f"[Scheduler Catch-up Error]: {e}")

    last_executed_slots = set()
    last_track_minute = -1

    while True:
        try:
            now = get_ist_now()
            today_str = now.strftime("%Y-%m-%d")
            hm_str = now.strftime("%H:%M")
            cur_minute = now.minute

            slot_850 = f"{today_str}_08:50"
            slot_915 = f"{today_str}_09:15"
            slot_1535 = f"{today_str}_15:35"
            slot_1700 = f"{today_str}_17:00"
            slot_catchup = f"{today_str}_catchup"

            # 0. Intraday Auto-Catchup Check (Handles PC sleep/wake during trading hours)
            is_trade, _ = is_trading_day(now)
            hm_val = now.hour * 100 + now.minute
            if is_trade and 915 <= hm_val <= 1530 and slot_catchup not in last_executed_slots:
                today_picks = PaperDB.get_picks_by_date(today_str)
                if not today_picks:
                    last_executed_slots.add(slot_catchup)
                    print(f"[Scheduler Auto-Catchup] Missing picks detected for trading day {today_str}. Running automated catch-up...")
                    try:
                        DailyPicker.run_daily_picker_catchup()
                    except Exception as e:
                        print(f"[Scheduler Auto-Catchup Error]: {e}")

            # 1. 08:50 AM Job
            if hm_str == "08:50" and slot_850 not in last_executed_slots:
                last_executed_slots.add(slot_850)
                job_morning_signals()

            # 2. 09:15 AM Job
            if hm_str == "09:15" and slot_915 not in last_executed_slots:
                last_executed_slots.add(slot_915)
                job_market_open_fills()

            # 3. 30-Minute Tracking Job (e.g. at minute 0 and 30 during market hours)
            if cur_minute in [0, 30] and cur_minute != last_track_minute:
                last_track_minute = cur_minute
                job_live_tracking()

            # 4. 03:35 PM EOD Evaluation Job
            if hm_str == "15:35" and slot_1535 not in last_executed_slots:
                last_executed_slots.add(slot_1535)
                job_eod_chronological_evaluation()

            # 5. Friday 05:00 PM Weekly Report Job
            if hm_str == "17:00" and slot_1700 not in last_executed_slots and now.weekday() == 4:
                last_executed_slots.add(slot_1700)
                job_weekly_report()

            time.sleep(10)
        except KeyboardInterrupt:
            print("\n[Scheduler] Exiting Paper Lab Scheduler.")
            break
        except Exception as e:
            print(f"[Scheduler] Error in loop: {e}")
            time.sleep(30)
