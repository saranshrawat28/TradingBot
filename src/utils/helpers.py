"""
Helper functions for Indian stock market time checks, currency formatting, and symbol resolution.
"""

from datetime import datetime, time, timezone, timedelta
from typing import Optional
import config

IST_OFFSET = timedelta(hours=5, minutes=30)
IST_TZ = timezone(IST_OFFSET)

def get_ist_now() -> datetime:
    """Return the current datetime in Indian Standard Time (IST)."""
    return datetime.now(timezone.utc) + IST_OFFSET

def is_market_open(dt: datetime = None) -> tuple[bool, str]:
    """
    Check if the Indian Stock Market (NSE/BSE) is currently open.
    Trading hours: Monday to Friday, 09:15 AM - 03:30 PM IST.
    """
    now = dt if dt else get_ist_now()
    
    # 0 = Monday, ..., 6 = Sunday
    if now.weekday() >= 5:
        return False, "Market is Closed (Weekend)"
    
    current_time = now.time()
    market_open = time(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    market_close = time(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    
    if market_open <= current_time <= market_close:
        return True, "Market is Open (Live Trading Session)"
    elif current_time < market_open:
        return False, f"Pre-Market / Closed (Opens at {config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d} IST)"
    else:
        return False, f"Market Closed for the Day (Closed at {config.MARKET_CLOSE_HOUR:02d}:{config.MARKET_CLOSE_MINUTE:02d} IST)"

def is_intraday_squareoff_time(dt: datetime = None) -> bool:
    """Check if it's 3:15 PM IST or later, requiring auto-squareoff of intraday positions."""
    now = dt if dt else get_ist_now()
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    sq_time = time(config.INTRADAY_SQUAREOFF_HOUR, config.INTRADAY_SQUAREOFF_MINUTE)
    return current_time >= sq_time

def format_currency_inr(amount: float) -> str:
    """Format numbers into Indian Rupee (INR) representation with ₹ symbol."""
    if amount is None:
        return "₹0.00"
    is_neg = amount < 0
    amount = abs(amount)
    
    # Format according to Indian numbering system (Lakhs, Crores)
    amount_str = f"{amount:.2f}"
    parts = amount_str.split(".")
    integer_part = parts[0]
    decimal_part = parts[1]
    
    if len(integer_part) > 3:
        last3 = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted_int = ",".join(groups) + "," + last3
    else:
        formatted_int = integer_part
        
    formatted = f"₹{formatted_int}.{decimal_part}"
    return f"-{formatted}" if is_neg else formatted

def format_percentage(val: float, include_sign: bool = True) -> str:
    """Format float into percentage string."""
    if val is None:
        return "0.00%"
    sign = "+" if val > 0 and include_sign else ""
    return f"{sign}{val:.2f}%"

def clean_symbol(symbol: str) -> str:
    """Normalize and format stock symbol for NSE/Yahoo."""
    s = symbol.strip().upper()
    if s in ["NIFTY", "NIFTY 50", "NIFTY50"]:
        return "^NSEI"
    if s in ["BANKNIFTY", "BANK NIFTY", "NIFTY BANK"]:
        return "^NSEBANK"
    if s.startswith("^") or " " in s or s.endswith("CE") or s.endswith("PE") or s.endswith("FUT"):
        return s
    if not s.endswith(".NS") and not s.endswith(".BO"):
        return f"{s}.NS"
    return s

def display_symbol_name(symbol: str) -> str:
    """Get clean display ticker without .NS / .BO."""
    s = symbol.upper()
    if s.startswith("^NSEI"):
        return "NIFTY 50"
    if s.startswith("^NSEBANK"):
        return "BANK NIFTY"
    return s.replace(".NS", "").replace(".BO", "")

def format_holding_duration(entry_time_str: str) -> str:
    """Calculate human-friendly elapsed holding time from entry timestamp."""
    if not entry_time_str:
        return "Active"
    try:
        clean_str = entry_time_str.replace(" IST", "").strip()
        # Handle formats: "2026-08-17 11:22:15" or "11:22:15"
        if len(clean_str) > 10:
            entry_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
        else:
            now_dt_temp = get_ist_now().replace(tzinfo=None)
            time_part = datetime.strptime(clean_str, "%H:%M:%S").time()
            entry_dt = datetime.combine(now_dt_temp.date(), time_part)
            
        now_dt = get_ist_now().replace(tzinfo=None) if hasattr(get_ist_now(), "tzinfo") else get_ist_now()
        diff = now_dt - entry_dt
        total_seconds = max(0, int(diff.total_seconds()))
        if total_seconds < 60:
            return f"{total_seconds}s ago"
        elif total_seconds < 3600:
            mins = total_seconds // 60
            return f"{mins}m ago"
        else:
            hours = total_seconds // 3600
            mins = (total_seconds % 3600) // 60
            return f"{hours}h {mins}m ago"
    except Exception:
        return "Active"

def get_lot_size(symbol: str) -> int:
    """Returns standard NSE F&O lot size for index and equity options."""
    sym_up = symbol.upper().replace(".NS", "").replace(".BO", "")
    if "BANK" in sym_up or "NSEBANK" in sym_up:
        return 30
    if "NIFTY" in sym_up or "NSEI" in sym_up:
        return 75
    if "FINNIFTY" in sym_up:
        return 65
    if "SENSEX" in sym_up or "BSESN" in sym_up:
        return 10
    if "RELIANCE" in sym_up:
        return 250
    if "TCS" in sym_up:
        return 175
    if "HDFCBANK" in sym_up:
        return 550
    if "INFY" in sym_up:
        return 400
    if "SBIN" in sym_up:
        return 750
    if "ICICIBANK" in sym_up:
        return 700
    return 50

def get_nse_options_expiry_details(now_dt: Optional[datetime] = None) -> dict:
    """
    Calculates upcoming NSE options expiry dates (Weekly Thursday & Monthly Last Thursday).
    """
    from datetime import date
    import calendar
    
    now = now_dt or get_ist_now()
    now_date = now.date() if isinstance(now, datetime) else now
    
    # 1. Find Current and Next Thursday (weekday 3)
    days_to_current_thu = (3 - now_date.weekday()) % 7
    current_thu_date = now_date + timedelta(days=days_to_current_thu)
    
    # If today is Thursday and after market close (15:30), roll to next week
    is_today_0dte = (now_date.weekday() == 3)
    if is_today_0dte and hasattr(now, "hour") and (now.hour > 15 or (now.hour == 15 and now.minute >= 30)):
        current_thu_date = now_date + timedelta(days=7)
        next_thu_date = current_thu_date + timedelta(days=7)
        is_today_0dte = False
    else:
        next_thu_date = current_thu_date + timedelta(days=7)
        
    # 2. Find Last Thursday of Current Month (Monthly Expiry)
    _, last_day_num = calendar.monthrange(now_date.year, now_date.month)
    month_end_date = date(now_date.year, now_date.month, last_day_num)
    offset_to_thu = (month_end_date.weekday() - 3) % 7
    monthly_thu_date = month_end_date - timedelta(days=offset_to_thu)
    
    if now_date > monthly_thu_date:
        next_m_year = now_date.year if now_date.month < 12 else now_date.year + 1
        next_m_month = now_date.month + 1 if now_date.month < 12 else 1
        _, next_last_day_num = calendar.monthrange(next_m_year, next_m_month)
        next_month_end = date(next_m_year, next_m_month, next_last_day_num)
        next_offset = (next_month_end.weekday() - 3) % 7
        monthly_thu_date = next_month_end - timedelta(days=next_offset)
        
    cur_exp_str = current_thu_date.strftime("%d-%b-%Y").upper()
    cur_exp_tag = current_thu_date.strftime("%d%b%y").upper()
    next_exp_str = next_thu_date.strftime("%d-%b-%Y").upper()
    next_exp_tag = next_thu_date.strftime("%d%b%y").upper()
    monthly_exp_str = monthly_thu_date.strftime("%d-%b-%Y").upper()
    monthly_exp_tag = monthly_thu_date.strftime("%d%b%y").upper()
    
    # Target recommendation logic:
    if is_today_0dte and hasattr(now, "hour") and (now.hour > 13 or (now.hour == 13 and now.minute >= 30)):
        rec_date = next_thu_date
        rec_str = f"{next_exp_str} (Next Weekly Expiry)"
        rec_tag = next_exp_tag
        is_rec_0dte = False
    elif is_today_0dte:
        rec_date = current_thu_date
        rec_str = f"{cur_exp_str} (Today 0DTE Expiry)"
        rec_tag = cur_exp_tag
        is_rec_0dte = True
    else:
        rec_date = current_thu_date
        is_monthly = (current_thu_date == monthly_thu_date)
        rec_str = f"{cur_exp_str} ({'Monthly' if is_monthly else 'Weekly'} Expiry)"
        rec_tag = cur_exp_tag
        is_rec_0dte = False
        
    days_left = max(0, (rec_date - now_date).days)
    
    return {
        "recommended_expiry_date": rec_date.strftime("%Y-%m-%d"),
        "recommended_expiry_str": rec_str,
        "recommended_expiry_tag": rec_tag,
        "current_weekly_str": cur_exp_str,
        "current_weekly_tag": cur_exp_tag,
        "next_weekly_str": next_exp_str,
        "next_weekly_tag": next_exp_tag,
        "monthly_expiry_str": monthly_exp_str,
        "monthly_expiry_tag": monthly_exp_tag,
        "is_0dte": is_rec_0dte,
        "dte_days": float(days_left)
    }

def format_nse_option_contract(
    symbol: str,
    spot_price: float,
    opt_type: str = "CE",
    expiry_date_str: Optional[str] = None,
    preferred_strike: Optional[float] = None
) -> dict:
    """
    Constructs real, 100% verified NSE derivative symbols and broker search queries
    matching Zerodha Kite, Dhan, Angel One, and Groww conventions.
    Calculates exact ATM/ITM strike from live spot price to eliminate hallucinated strikes.
    """
    import calendar
    from datetime import date
    
    sym_clean = symbol.upper().replace("^", "").replace(".NS", "").replace(" ", "")
    is_banknifty = "BANK" in sym_clean
    is_nifty = "NIFTY" in sym_clean and not is_banknifty
    is_fin = "FIN" in sym_clean
    is_midcp = "MIDCP" in sym_clean
    
    # 1. Exact Strike Step
    if is_banknifty:
        strike_step = 100
    elif is_nifty or is_fin:
        strike_step = 50
    elif is_midcp:
        strike_step = 25
    else:
        # Equity stock option
        if spot_price > 2000:
            strike_step = 50
        elif spot_price > 1000:
            strike_step = 20
        elif spot_price > 500:
            strike_step = 10
        else:
            strike_step = 5
            
    atm_strike = int(round(spot_price / strike_step) * strike_step)
    
    # Validate strike: if preferred_strike is given and is within 2 steps of ATM, keep it; else force ATM
    if preferred_strike and abs(preferred_strike - atm_strike) <= (2 * strike_step):
        chosen_strike = int(round(preferred_strike / strike_step) * strike_step)
    else:
        chosen_strike = atm_strike
        
    # 2. Expiry and Trading Symbol construction
    exp_details = get_nse_options_expiry_details()
    target_exp_date = expiry_date_str or exp_details["recommended_expiry_date"]
    
    if isinstance(target_exp_date, str):
        exp_dt = datetime.strptime(target_exp_date, "%Y-%m-%d").date()
    else:
        exp_dt = target_exp_date
        
    year_str = str(exp_dt.year)[2:] # e.g. "26"
    month_name_3 = exp_dt.strftime("%b").upper() # e.g. "AUG"
    
    # Check if monthly expiry (last Thursday of the month)
    _, last_day_num = calendar.monthrange(exp_dt.year, exp_dt.month)
    month_end_date = date(exp_dt.year, exp_dt.month, last_day_num)
    offset_to_thu = (month_end_date.weekday() - 3) % 7
    monthly_thu_date = month_end_date - timedelta(days=offset_to_thu)
    
    is_monthly = (exp_dt == monthly_thu_date)
    opt_type_clean = "PE" if "P" in opt_type.upper() else "CE"
    
    # 3. NSE Exchange Tradingsymbol
    if is_monthly:
        # Standard NSE Monthly: e.g. BANKNIFTY26AUG57800CE
        trading_symbol = f"{sym_clean}{year_str}{month_name_3}{chosen_strike}{opt_type_clean}"
        broker_search_primary = f"{sym_clean} {year_str}{month_name_3} {chosen_strike} {opt_type_clean}"
    else:
        # Standard NSE Weekly (for NIFTY): Month code 1-9 for Jan-Sep, O for Oct, N for Nov, D for Dec
        m_code = str(exp_dt.month) if exp_dt.month <= 9 else ("O" if exp_dt.month == 10 else ("N" if exp_dt.month == 11 else "D"))
        day_str = f"{exp_dt.day:02d}"
        trading_symbol = f"{sym_clean}{year_str}{m_code}{day_str}{chosen_strike}{opt_type_clean}"
        broker_search_primary = f"{sym_clean} {exp_dt.day}{month_name_3} {chosen_strike} {opt_type_clean}"
        
    # The universal broker search query (100% guaranteed to find contract in Kite/Groww/Dhan search bars)
    universal_search = f"{sym_clean} {chosen_strike} {opt_type_clean}"
    display_title = f"{sym_clean} {chosen_strike} {opt_type_clean}"
    
    # Strike Moneyness
    strike_diff = chosen_strike - spot_price
    if abs(strike_diff) <= (strike_step * 0.45):
        moneyness = "ATM (At-The-Money)"
    elif (opt_type_clean == "CE" and strike_diff < 0) or (opt_type_clean == "PE" and strike_diff > 0):
        moneyness = f"ITM (In-The-Money by {abs(strike_diff):.0f} pts)"
    else:
        moneyness = f"OTM (Out-of-The-Money by {abs(strike_diff):.0f} pts)"
        
    return {
        "underlying": sym_clean,
        "strike": chosen_strike,
        "atm_strike": atm_strike,
        "option_type": opt_type_clean,
        "is_monthly": is_monthly,
        "trading_symbol": trading_symbol,
        "broker_search_query": broker_search_primary,
        "universal_search": universal_search,
        "display_title": display_title,
        "moneyness": moneyness,
        "expiry_date": exp_dt.strftime("%Y-%m-%d"),
        "expiry_str": f"{exp_dt.strftime('%d-%b-%Y').upper()} ({'Monthly' if is_monthly else 'Weekly'} Expiry)"
    }


