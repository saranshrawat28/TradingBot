"""
NSE & BSE Trading Holiday Calendar and Market Schedule Verifier.
"""

from datetime import datetime, date
from typing import Tuple, Set

# Indian Gazetted Stock Exchange Trading Holidays (NSE/BSE Equities)
# Covers 2025 and 2026 typical gazetted dates
NSE_HOLIDAYS: Set[str] = {
    # 2025
    "2025-01-26", # Republic Day
    "2025-02-26", # Mahashivratri
    "2025-03-14", # Holi
    "2025-03-31", # Id-Ul-Fitr
    "2025-04-10", # Mahavir Jayanti
    "2025-04-14", # Dr. Baba Saheb Ambedkar Jayanti
    "2025-04-18", # Good Friday
    "2025-05-01", # Maharashtra Day
    "2025-06-07", # Bakri Id
    "2025-08-15", # Independence Day
    "2025-08-27", # Ganesh Chaturthi
    "2025-10-02", # Mahatma Gandhi Jayanti
    "2025-10-21", # Diwali Laxmi Pujan
    "2025-10-22", # Diwali Balipratipada
    "2025-11-05", # Gurunanak Jayanti
    "2025-12-25", # Christmas
    # 2026
    "2026-01-26", # Republic Day
    "2026-02-16", # Mahashivratri
    "2026-03-03", # Holi
    "2026-03-20", # Id-Ul-Fitr
    "2026-04-03", # Good Friday
    "2026-04-14", # Dr. Ambedkar Jayanti
    "2026-05-01", # Maharashtra Day
    "2026-05-27", # Bakri Id
    "2026-08-15", # Independence Day
    "2026-09-15", # Ganesh Chaturthi
    "2026-10-02", # Gandhi Jayanti
    "2026-10-20", # Dussehra
    "2026-11-08", # Diwali
    "2026-11-24", # Gurunanak Jayanti
    "2026-12-25", # Christmas
}

def is_trading_day(dt: datetime = None) -> Tuple[bool, str]:
    """
    Checks if a given datetime is an active NSE/BSE trading day.
    Returns (is_trading, reason).
    """
    if dt is None:
        from src.utils.helpers import get_ist_now
        dt = get_ist_now()

    d = dt.date() if isinstance(dt, datetime) else dt
    d_str = d.strftime("%Y-%m-%d")

    # Check Weekend (5 = Saturday, 6 = Sunday)
    if d.weekday() == 5:
        return False, f"Market Closed: Saturday Weekend ({d_str})"
    if d.weekday() == 6:
        return False, f"Market Closed: Sunday Weekend ({d_str})"

    # Check Public Holidays
    if d_str in NSE_HOLIDAYS:
        return False, f"Market Closed: NSE Public Holiday ({d_str})"

    return True, f"Active Trading Day ({d_str})"
