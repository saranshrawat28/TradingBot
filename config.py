"""
Configuration settings for Indian Stocks & F&O Algorithmic Trading Bot
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cache"
STORAGE_DIR = BASE_DIR / "storage"

DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# Comprehensive Indian Market Universe (NSE Equities, Indices & Popular Stocks)
# -------------------------------------------------------------
DEFAULT_WATCHLIST = [
    # Major Indices
    {"symbol": "^NSEI", "name": "NIFTY 50", "category": "Indices"},
    {"symbol": "^NSEBANK", "name": "BANK NIFTY", "category": "Indices"},
    {"symbol": "^CNXIT", "name": "NIFTY IT", "category": "Indices"},
    
    # Banking & Financial Services
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "category": "Banking"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "category": "Banking"},
    {"symbol": "SBIN.NS", "name": "State Bank of India (SBI)", "category": "Banking"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "category": "Banking"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank", "category": "Banking"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank", "category": "Banking"},
    {"symbol": "FEDERALBNK.NS", "name": "Federal Bank", "category": "Banking"},
    {"symbol": "IDFCFIRSTB.NS", "name": "IDFC First Bank", "category": "Banking"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "category": "Financial Services"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "category": "Financial Services"},
    {"symbol": "JIOFIN.NS", "name": "Jio Financial Services", "category": "Financial Services"},
    {"symbol": "SHRIRAMFIN.NS", "name": "Shriram Finance", "category": "Financial Services"},
    {"symbol": "RECLTD.NS", "name": "REC Ltd (Rural Electrification)", "category": "Financial Services"},
    {"symbol": "PFC.NS", "name": "Power Finance Corporation", "category": "Financial Services"},
    
    # IT & Tech
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services (TCS)", "category": "IT & Tech"},
    {"symbol": "INFY.NS", "name": "Infosys", "category": "IT & Tech"},
    {"symbol": "WIPRO.NS", "name": "Wipro", "category": "IT & Tech"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "category": "IT & Tech"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra", "category": "IT & Tech"},
    {"symbol": "LTIM.NS", "name": "LTIMindtree", "category": "IT & Tech"},
    {"symbol": "PERSISTENT.NS", "name": "Persistent Systems", "category": "IT & Tech"},
    {"symbol": "COFORGE.NS", "name": "Coforge", "category": "IT & Tech"},
    {"symbol": "ETERNAL.NS", "name": "Zomato (Eternal Ltd)", "category": "New Age / Internet"},
    {"symbol": "PAYTM.NS", "name": "Paytm (One97 Communications)", "category": "New Age / Internet"},
    
    # Energy, Oil & Power
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "category": "Energy & Conglomerate"},
    {"symbol": "NTPC.NS", "name": "NTPC Ltd (Power Generation)", "category": "Power & Energy"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation", "category": "Power & Energy"},
    {"symbol": "ONGC.NS", "name": "ONGC (Oil & Natural Gas)", "category": "Oil & Gas"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum (BPCL)", "category": "Oil & Gas"},
    {"symbol": "IOC.NS", "name": "Indian Oil Corporation (IOC)", "category": "Oil & Gas"},
    {"symbol": "TATAPOWER.NS", "name": "Tata Power", "category": "Power & Energy"},
    {"symbol": "ADANIGREEN.NS", "name": "Adani Green Energy", "category": "Power & Energy"},
    {"symbol": "SUZLON.NS", "name": "Suzlon Energy (Wind Power)", "category": "Power & Energy"},
    {"symbol": "COALINDIA.NS", "name": "Coal India", "category": "Mining & Energy"},
    
    # Automobile & EV
    {"symbol": "TMCV.NS", "name": "Tata Motors Ltd", "category": "Automobile"},
    {"symbol": "TMPV.NS", "name": "Tata Motors Passenger Vehicles", "category": "Automobile"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki", "category": "Automobile"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra", "category": "Automobile"},
    {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto", "category": "Automobile"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors (Royal Enfield)", "category": "Automobile"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp", "category": "Automobile"},
    {"symbol": "TVSMOTOR.NS", "name": "TVS Motor Company", "category": "Automobile"},
    
    # Defence, Railways & Infrastructure
    {"symbol": "LT.NS", "name": "Larsen & Toubro (L&T)", "category": "Infrastructure"},
    {"symbol": "HAL.NS", "name": "Hindustan Aeronautics (HAL)", "category": "Defence & Aerospace"},
    {"symbol": "BEL.NS", "name": "Bharat Electronics (BEL)", "category": "Defence & Electronics"},
    {"symbol": "MAZDOCK.NS", "name": "Mazagon Dock Shipbuilders", "category": "Defence & Shipbuilding"},
    {"symbol": "COCHINSHIP.NS", "name": "Cochin Shipyard", "category": "Defence & Shipbuilding"},
    {"symbol": "IRFC.NS", "name": "Indian Railway Finance (IRFC)", "category": "Railways"},
    {"symbol": "RVNL.NS", "name": "Rail Vikas Nigam (RVNL)", "category": "Railways"},
    {"symbol": "IRCTC.NS", "name": "IRCTC (Indian Railways)", "category": "Railways & Tourism"},
    {"symbol": "BHEL.NS", "name": "Bharat Heavy Electricals (BHEL)", "category": "Capital Goods"},
    {"symbol": "SIEMENS.NS", "name": "Siemens India", "category": "Capital Goods"},
    {"symbol": "ABB.NS", "name": "ABB India", "category": "Capital Goods"},
    {"symbol": "DIXON.NS", "name": "Dixon Technologies", "category": "Electronics Manufacturing"},
    
    # FMCG & Consumer Goods
    {"symbol": "ITC.NS", "name": "ITC Ltd", "category": "FMCG"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever (HUL)", "category": "FMCG"},
    {"symbol": "NESTLEIND.NS", "name": "Nestlé India", "category": "FMCG"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries", "category": "FMCG"},
    {"symbol": "TATACONSUM.NS", "name": "Tata Consumer Products", "category": "FMCG"},
    {"symbol": "VBL.NS", "name": "Varun Beverages (Pepsi Bottler)", "category": "FMCG"},
    {"symbol": "GODREJCP.NS", "name": "Godrej Consumer Products", "category": "FMCG"},
    {"symbol": "TITAN.NS", "name": "Titan Company (Tanishq/Fastrack)", "category": "Consumer Retail"},
    {"symbol": "TRENT.NS", "name": "Trent (Zudio / Westside)", "category": "Consumer Retail"},
    {"symbol": "KALYANKJIL.NS", "name": "Kalyan Jewellers", "category": "Consumer Retail"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "category": "Paints & Home"},
    
    # Metals, Cement & Commodities
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "category": "Metals & Mining"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel", "category": "Metals & Mining"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco Industries (Aluminium)", "category": "Metals & Mining"},
    {"symbol": "VEDL.NS", "name": "Vedanta Ltd", "category": "Metals & Mining"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement", "category": "Cement & Building"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries", "category": "Cement & Chemicals"},
    
    # Pharmaceuticals & Healthcare
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma", "category": "Healthcare & Pharma"},
    {"symbol": "CIPLA.NS", "name": "Cipla", "category": "Healthcare & Pharma"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories", "category": "Healthcare & Pharma"},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories", "category": "Healthcare & Pharma"},
    {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals", "category": "Healthcare & Hospitals"},
    
    # Telecom & Adani Group
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "category": "Telecom"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "category": "Conglomerate"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "category": "Ports & Logistics"},
    {"symbol": "DLF.NS", "name": "DLF Ltd", "category": "Real Estate"},
    
    # New Listings, Recent IPOs & High-Growth Momentum
    {"symbol": "SWIGGY.NS", "name": "Swiggy Ltd", "category": "New Listings & IPOs"},
    {"symbol": "HYUNDAI.NS", "name": "Hyundai Motor India", "category": "New Listings & IPOs"},
    {"symbol": "BAJAJHFL.NS", "name": "Bajaj Housing Finance", "category": "New Listings & IPOs"},
    {"symbol": "WAAREEENER.NS", "name": "Waaree Energies", "category": "New Listings & IPOs"},
    {"symbol": "PREMIERENE.NS", "name": "Premier Energies", "category": "New Listings & IPOs"},
    {"symbol": "NTPCGREEN.NS", "name": "NTPC Green Energy", "category": "New Listings & IPOs"},
    {"symbol": "TATATECH.NS", "name": "Tata Technologies", "category": "New Listings & IPOs"},
    {"symbol": "IREDA.NS", "name": "IREDA", "category": "New Listings & IPOs"},
    {"symbol": "OLAELC.NS", "name": "Ola Electric Mobility", "category": "New Listings & IPOs"},
    {"symbol": "MANKIND.NS", "name": "Mankind Pharma", "category": "New Listings & IPOs"},
    {"symbol": "KAYNES.NS", "name": "Kaynes Technology", "category": "New Listings & IPOs"},
]

def load_watchlist() -> list[dict]:
    """Load watchlist from external data_cache/watchlist.json if present, otherwise DEFAULT_WATCHLIST."""
    import json
    json_path = DATA_DIR / "watchlist.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list) and len(loaded) > 0:
                    return loaded
        except Exception:
            pass
    return DEFAULT_WATCHLIST

# Quick symbol list
POPULAR_SYMBOLS = [item["symbol"] for item in load_watchlist()]

# -------------------------------------------------------------
# Default Risk Management & Capital Settings
# -------------------------------------------------------------
DEFAULT_INITIAL_CAPITAL = 100000.0  # ₹1,00,000 INR
DEFAULT_RISK_PER_TRADE_PCT = 2.0     # 2% maximum equity risk per trade
DEFAULT_STOP_LOSS_PCT = 1.5          # 1.5% default stop loss
DEFAULT_TAKE_PROFIT_PCT = 3.0        # 3.0% default take profit (1:2 Risk/Reward)
DEFAULT_TRAILING_SL_PCT = 1.0        # 1.0% trailing stop loss trigger
MAX_DAILY_LOSS_PCT = 5.0             # 5% max daily drawdown circuit breaker
MAX_CONCURRENT_POSITIONS = 5

# Indian Market Timings (IST: UTC + 5:30)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
INTRADAY_SQUAREOFF_HOUR = 15
INTRADAY_SQUAREOFF_MINUTE = 15

# -------------------------------------------------------------
# Indian Brokerage & Regulatory Cost Rules (NSE Intraday / Delivery)
# -------------------------------------------------------------
INDIAN_FEES = {
    "brokerage_per_order": 20.0,       # Max ₹20 per executed order (Discount Broker standard)
    "brokerage_pct": 0.0003,           # 0.03% or ₹20 whichever is lower
    "stt_intraday_sell_pct": 0.00025,  # 0.025% STT on sell for Intraday (Equity)
    "stt_delivery_pct": 0.001,         # 0.1% STT on buy & sell for Delivery
    "exchange_txn_charge_pct": 0.0000345, # 0.00345% NSE turnover charge
    "sebi_turnover_pct": 0.000001,     # ₹10 per crore (0.0001%)
    "stamp_duty_buy_pct": 0.00003,     # 0.003% on buy
    "gst_pct": 0.18,                   # 18% on (Brokerage + Txn Charges + SEBI)
    "slippage_pct": 0.0005,            # 0.05% realistic execution slippage
}

# -------------------------------------------------------------
# Broker API Credentials (Loaded from environment variables)
# -------------------------------------------------------------
ZERODHA_API_KEY = os.getenv("ZERODHA_API_KEY", "")
ZERODHA_API_SECRET = os.getenv("ZERODHA_API_SECRET", "")
ZERODHA_ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN", "")

ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PIN = os.getenv("ANGEL_PIN", "")
ANGEL_TOTP_KEY = os.getenv("ANGEL_TOTP_KEY", "")

DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")

ACTIVE_BROKER = os.getenv("ACTIVE_BROKER", "paper")
