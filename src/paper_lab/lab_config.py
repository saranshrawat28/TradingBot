"""
Configuration and versioned threshold parameters for ApexTrade Paper Trading Lab.
"""

from typing import List

class LabConfig:
    # Version stamp embedded in every pick and report
    CONFIG_VERSION = "v1.0.0"

    # Core Screening & Sizing Parameters
    PICKS_PER_DAY: int = 5
    DAILY_CAPITAL_PER_PICK: float = 20000.0   # Fixed notional per pick (₹20,000 * 5 = ₹1,00,000/day)
    TOTAL_DAILY_CAPITAL: float = 100000.0

    # Stock Selection Thresholds
    MIN_ADVISOR_SCORE: float = 6.5
    VALID_VERDICTS: List[str] = ["BUY", "STRONG_BUY", "BUY_ON_PULLBACK"]

    # Diagnostic Indicator Thresholds (used in signal failure analysis)
    MAX_ENTRY_RSI_THRESHOLD: float = 65.0      # Entries above this are flagged for RSI exhaustion
    MIN_ENTRY_RVOL_THRESHOLD: float = 1.00     # Entries below this are flagged for weak volume
    MAX_VWAP_SIGMA_THRESHOLD: float = 0.40     # Entries > 0.4 sigma above VWAP are flagged as chase/late
    MIN_ADX_TREND_THRESHOLD: float = 20.0      # Entries below 20 are flagged as ranging chop

    # Risk & Reward Defaults (if ATR unavailable)
    DEFAULT_T1_GAIN_PCT: float = 2.5
    DEFAULT_T2_GAIN_PCT: float = 5.0
    DEFAULT_SL_LOSS_PCT: float = 1.5

    # Evaluation Timings (IST 24h format)
    SIGNAL_GEN_TIME_STR: str = "08:50"
    MARKET_OPEN_TIME_STR: str = "09:15"
    EOD_EVAL_TIME_STR: str = "15:35"
    WEEKLY_REPORT_TIME_STR: str = "17:00"

    # Universe of liquid Indian equities scanned for morning momentum
    UNIVERSE: List[str] = [
        "RELIANCE.NS",
        "TMCV.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "TCS.NS",
        "BHARTIARTL.NS",
        "LT.NS",
        "ETERNAL.NS",
        "M&M.NS",
        "SUNPHARMA.NS",
        "BAJFINANCE.NS",
        "AXISBANK.NS",
        "TITAN.NS",
        "ITC.NS",
        "WIPRO.NS",
        "COALINDIA.NS",
        "HINDALCO.NS",
        "TATASTEEL.NS",
        "TATAPOWER.NS",
        "ADANIENT.NS",
        "ADANIPORTS.NS",
        "JIOFIN.NS",
        "HAL.NS",
        "BEL.NS",
        "IRFC.NS",
        "RVNL.NS",
        "SUZLON.NS",
        "PAYTM.NS",
        "NTPC.NS",
        "POWERGRID.NS",
        "ONGC.NS",
        "BPCL.NS",
        "MARUTI.NS",
        "KOTAKBANK.NS",
        "VEDL.NS",
        "DLF.NS",
        "TRENT.NS",
        "HDFCLIFE.NS",
        "SWIGGY.NS",
        "HYUNDAI.NS",
        "BAJAJHFL.NS",
        "WAAREEENER.NS",
        "TATATECH.NS",
        "IREDA.NS"
    ]
