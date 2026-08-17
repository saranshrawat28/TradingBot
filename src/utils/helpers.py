"""
Helper functions for Indian stock market time checks, currency formatting, and symbol resolution.
"""

from datetime import datetime, time, timezone, timedelta
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

