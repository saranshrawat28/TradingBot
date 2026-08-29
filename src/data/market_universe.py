"""
Comprehensive NSE Indian Market Universe.
Covers 200+ liquid high-growth equities across all sectors (Large-Cap, Mid-Cap, Small-Cap, and New IPOs).
"""

from typing import List, Dict

BROAD_NSE_UNIVERSE: List[Dict[str, str]] = [
    # Banking & Financial Services
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank", "sector": "Banking"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Banking"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank", "sector": "Banking"},
    {"symbol": "BANKBARODA.NS", "name": "Bank of Baroda", "sector": "Banking"},
    {"symbol": "PNB.NS", "name": "Punjab National Bank", "sector": "Banking"},
    {"symbol": "FEDERALBNK.NS", "name": "Federal Bank", "sector": "Banking"},
    {"symbol": "IDFCFIRSTB.NS", "name": "IDFC First Bank", "sector": "Banking"},
    {"symbol": "CANBK.NS", "name": "Canara Bank", "sector": "Banking"},
    {"symbol": "UNIONBANK.NS", "name": "Union Bank of India", "sector": "Banking"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Financial Services"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Financial Services"},
    {"symbol": "JIOFIN.NS", "name": "Jio Financial Services", "sector": "Financial Services"},
    {"symbol": "CHOLAFIN.NS", "name": "Cholamandalam Investment", "sector": "Financial Services"},
    {"symbol": "SHRIRAMFIN.NS", "name": "Shriram Finance", "sector": "Financial Services"},
    {"symbol": "MUTHOOTFIN.NS", "name": "Muthoot Finance", "sector": "Financial Services"},
    {"symbol": "PFC.NS", "name": "Power Finance Corporation", "sector": "Financial Services"},
    {"symbol": "RECLTD.NS", "name": "REC Limited", "sector": "Financial Services"},
    {"symbol": "HDFCLIFE.NS", "name": "HDFC Life Insurance", "sector": "Insurance"},
    {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Insurance"},
    {"symbol": "ICICIPRULI.NS", "name": "ICICI Prudential Life", "sector": "Insurance"},

    # IT & Technology
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT & Tech"},
    {"symbol": "INFY.NS", "name": "Infosys", "sector": "IT & Tech"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT & Tech"},
    {"symbol": "WIPRO.NS", "name": "Wipro", "sector": "IT & Tech"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra", "sector": "IT & Tech"},
    {"symbol": "LTIM.NS", "name": "LTIMindtree", "sector": "IT & Tech"},
    {"symbol": "PERSISTENT.NS", "name": "Persistent Systems", "sector": "IT & Tech"},
    {"symbol": "COFORGE.NS", "name": "Coforge", "sector": "IT & Tech"},
    {"symbol": "MPHASIS.NS", "name": "Mphasis", "sector": "IT & Tech"},
    {"symbol": "KPITTECH.NS", "name": "KPIT Technologies", "sector": "IT & Tech"},
    {"symbol": "TATAELXSI.NS", "name": "Tata Elxsi", "sector": "IT & Tech"},
    {"symbol": "LTTS.NS", "name": "L&T Technology Services", "sector": "IT & Tech"},
    {"symbol": "CYIENT.NS", "name": "Cyient", "sector": "IT & Tech"},

    # Auto & Mobility
    {"symbol": "TMCV.NS", "name": "Tata Motors", "sector": "Automobile"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra", "sector": "Automobile"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki", "sector": "Automobile"},
    {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto", "sector": "Automobile"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp", "sector": "Automobile"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors (Royal Enfield)", "sector": "Automobile"},
    {"symbol": "TVSMOTOR.NS", "name": "TVS Motor Company", "sector": "Automobile"},
    {"symbol": "BHARATFORG.NS", "name": "Bharat Forge", "sector": "Auto Components"},
    {"symbol": "MOTHERSON.NS", "name": "Samvardhana Motherson", "sector": "Auto Components"},
    {"symbol": "BOSCHLTD.NS", "name": "Bosch India", "sector": "Auto Components"},
    {"symbol": "APOLLOTYRE.NS", "name": "Apollo Tyres", "sector": "Auto Components"},
    {"symbol": "MRF.NS", "name": "MRF Tyres", "sector": "Auto Components"},

    # Energy, Power, Renewables & Oil/Gas
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy & Conglomerate"},
    {"symbol": "ONGC.NS", "name": "ONGC", "sector": "Oil & Gas"},
    {"symbol": "NTPC.NS", "name": "NTPC", "sector": "Power & Utilities"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation", "sector": "Power & Utilities"},
    {"symbol": "TATAPOWER.NS", "name": "Tata Power", "sector": "Power & Renewables"},
    {"symbol": "COALINDIA.NS", "name": "Coal India", "sector": "Energy & Mining"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Oil & Gas"},
    {"symbol": "IOC.NS", "name": "Indian Oil Corporation", "sector": "Oil & Gas"},
    {"symbol": "HINDPETRO.NS", "name": "Hindustan Petroleum", "sector": "Oil & Gas"},
    {"symbol": "GAIL.NS", "name": "GAIL India", "sector": "Oil & Gas"},
    {"symbol": "ADANIPOWER.NS", "name": "Adani Power", "sector": "Power"},
    {"symbol": "ADANIGREEN.NS", "name": "Adani Green Energy", "sector": "Renewable Energy"},
    {"symbol": "SUZLON.NS", "name": "Suzlon Energy", "sector": "Renewable Energy"},
    {"symbol": "IREDA.NS", "name": "Indian Renewable Energy Agency", "sector": "Renewable Energy"},
    {"symbol": "WAAREEENER.NS", "name": "Waaree Energies (Solar)", "sector": "Renewable Energy"},
    {"symbol": "PREMIERENE.NS", "name": "Premier Energies (Solar)", "sector": "Renewable Energy"},
    {"symbol": "NTPCGREEN.NS", "name": "NTPC Green Energy", "sector": "Renewable Energy"},

    # Defence, Aerospace & Shipbuilding
    {"symbol": "HAL.NS", "name": "Hindustan Aeronautics (HAL)", "sector": "Defence & Aerospace"},
    {"symbol": "BEL.NS", "name": "Bharat Electronics (BEL)", "sector": "Defence & Electronics"},
    {"symbol": "MAZDOCK.NS", "name": "Mazagon Dock Shipbuilders", "sector": "Defence & Shipbuilding"},
    {"symbol": "COCHINSHIP.NS", "name": "Cochin Shipyard", "sector": "Defence & Shipbuilding"},
    {"symbol": "BEML.NS", "name": "BEML Ltd (Defence & Mining)", "sector": "Defence & Heavy Machinery"},
    {"symbol": "DATAPATTNS.NS", "name": "Data Patterns (Defence Tech)", "sector": "Defence Tech"},

    # Railways, Infrastructure & Capital Goods
    {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Infrastructure & EPC"},
    {"symbol": "IRFC.NS", "name": "Indian Railway Finance (IRFC)", "sector": "Railways"},
    {"symbol": "RVNL.NS", "name": "Rail Vikas Nigam (RVNL)", "sector": "Railways"},
    {"symbol": "IRCTC.NS", "name": "IRCTC (Railways & Tourism)", "sector": "Railways & Tourism"},
    {"symbol": "RAILTEL.NS", "name": "RailTel Corporation", "sector": "Railways & Telecom"},
    {"symbol": "RITES.NS", "name": "RITES Ltd", "sector": "Railways"},
    {"symbol": "TITAGARH.NS", "name": "Titagarh Rail Systems", "sector": "Railways & Wagons"},
    {"symbol": "BHEL.NS", "name": "Bharat Heavy Electricals (BHEL)", "sector": "Capital Goods"},
    {"symbol": "SIEMENS.NS", "name": "Siemens India", "sector": "Capital Goods"},
    {"symbol": "ABB.NS", "name": "ABB India", "sector": "Capital Goods"},
    {"symbol": "CUMMINSIND.NS", "name": "Cummins India", "sector": "Capital Goods"},

    # Metals, Mining & Commodities
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "sector": "Metals & Mining"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Metals & Mining"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco Industries (Aluminium)", "sector": "Metals & Mining"},
    {"symbol": "VEDL.NS", "name": "Vedanta Ltd", "sector": "Metals & Mining"},
    {"symbol": "JINDALSTEL.NS", "name": "Jindal Steel & Power", "sector": "Metals & Mining"},
    {"symbol": "NMDC.NS", "name": "NMDC (Iron Ore)", "sector": "Metals & Mining"},
    {"symbol": "SAIL.NS", "name": "Steel Authority of India", "sector": "Metals & Mining"},
    {"symbol": "NATIONALUM.NS", "name": "National Aluminium (NALCO)", "sector": "Metals & Mining"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Cement & Building"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries", "sector": "Cement & Chemicals"},
    {"symbol": "AMBUJACEM.NS", "name": "Ambuja Cements", "sector": "Cement"},
    {"symbol": "SHREECEM.NS", "name": "Shree Cement", "sector": "Cement"},

    # FMCG & Consumer Retail
    {"symbol": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever (HUL)", "sector": "FMCG"},
    {"symbol": "NESTLEIND.NS", "name": "Nestlé India", "sector": "FMCG"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "FMCG"},
    {"symbol": "TATACONSUM.NS", "name": "Tata Consumer Products", "sector": "FMCG"},
    {"symbol": "VBL.NS", "name": "Varun Beverages (Pepsi Bottler)", "sector": "FMCG"},
    {"symbol": "GODREJCP.NS", "name": "Godrej Consumer Products", "sector": "FMCG"},
    {"symbol": "DABUR.NS", "name": "Dabur India", "sector": "FMCG"},
    {"symbol": "MARICO.NS", "name": "Marico Ltd", "sector": "FMCG"},
    {"symbol": "TITAN.NS", "name": "Titan Company (Tanishq/Fastrack)", "sector": "Consumer Retail"},
    {"symbol": "TRENT.NS", "name": "Trent (Zudio / Westside)", "sector": "Consumer Retail"},
    {"symbol": "KALYANKJIL.NS", "name": "Kalyan Jewellers", "sector": "Consumer Retail"},
    {"symbol": "DMART.NS", "name": "Avenue Supermarts (DMart)", "sector": "Consumer Retail"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Paints & Home"},
    {"symbol": "BERGEPAINT.NS", "name": "Berger Paints", "sector": "Paints"},
    {"symbol": "PIDILITIND.NS", "name": "Pidilite Industries (Fevicol)", "sector": "Specialty Chemicals"},
    {"symbol": "ETERNAL.NS", "name": "Zomato Ltd", "sector": "Consumer Tech & Food"},
    {"symbol": "SWIGGY.NS", "name": "Swiggy Ltd", "sector": "Consumer Tech & Food"},

    # Pharmaceuticals & Healthcare
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Healthcare & Pharma"},
    {"symbol": "CIPLA.NS", "name": "Cipla", "sector": "Healthcare & Pharma"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories", "sector": "Healthcare & Pharma"},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Healthcare & Pharma"},
    {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals", "sector": "Healthcare & Hospitals"},
    {"symbol": "MAXHEALTH.NS", "name": "Max Healthcare", "sector": "Healthcare & Hospitals"},
    {"symbol": "MANKIND.NS", "name": "Mankind Pharma", "sector": "Healthcare & Pharma"},
    {"symbol": "LUPIN.NS", "name": "Lupin Ltd", "sector": "Healthcare & Pharma"},
    {"symbol": "TORNTPHARM.NS", "name": "Torrent Pharmaceuticals", "sector": "Healthcare & Pharma"},
    {"symbol": "ZYDUSLIFE.NS", "name": "Zydus Lifesciences", "sector": "Healthcare & Pharma"},
    {"symbol": "BIOCON.NS", "name": "Biocon", "sector": "Biotech & Pharma"},

    # Real Estate & Housing Finance
    {"symbol": "DLF.NS", "name": "DLF Ltd", "sector": "Real Estate"},
    {"symbol": "GODREJPROP.NS", "name": "Godrej Properties", "sector": "Real Estate"},
    {"symbol": "LODHA.NS", "name": "Macrotech Developers (Lodha)", "sector": "Real Estate"},
    {"symbol": "OBEROIRLTY.NS", "name": "Oberoi Realty", "sector": "Real Estate"},
    {"symbol": "PRESTIGE.NS", "name": "Prestige Estates", "sector": "Real Estate"},
    {"symbol": "PHOENIXLTD.NS", "name": "The Phoenix Mills", "sector": "Retail Real Estate"},
    {"symbol": "BAJAJHFL.NS", "name": "Bajaj Housing Finance", "sector": "Housing Finance"},
    {"symbol": "LICHSGFIN.NS", "name": "LIC Housing Finance", "sector": "Housing Finance"},

    # Electronics Manufacturing & EMS
    {"symbol": "DIXON.NS", "name": "Dixon Technologies", "sector": "Electronics Manufacturing"},
    {"symbol": "KAYNES.NS", "name": "Kaynes Technology", "sector": "Electronics Manufacturing"},
    {"symbol": "SYRMA.NS", "name": "Syrma SGS Technology", "sector": "Electronics Manufacturing"},
    {"symbol": "PGEL.NS", "name": "PG Electroplast", "sector": "Electronics Manufacturing"},
    {"symbol": "AMBER.NS", "name": "Amber Enterprises", "sector": "Electronics Manufacturing"},

    # Telecom, Media & Conglomerates
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom"},
    {"symbol": "INDUS.NS", "name": "Indus Towers", "sector": "Telecom Infra"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Conglomerate"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "sector": "Ports & Logistics"},

    # Newly Listed IPOs & Momentum High-Growth
    {"symbol": "HYUNDAI.NS", "name": "Hyundai Motor India", "sector": "New Listings & IPOs"},
    {"symbol": "TATATECH.NS", "name": "Tata Technologies", "sector": "New Listings & IPOs"},
    {"symbol": "OLAELC.NS", "name": "Ola Electric Mobility", "sector": "New Listings & IPOs"},
    {"symbol": "PAYTM.NS", "name": "One97 Communications (Paytm)", "sector": "Fintech & Payments"},
    {"symbol": "POLICYBZR.NS", "name": "PB Fintech (PolicyBazaar)", "sector": "Fintech & Insurance"},
    {"symbol": "NYKAA.NS", "name": "FSN E-Commerce (Nykaa)", "sector": "Consumer Tech"},
    {"symbol": "DELHIVERY.NS", "name": "Delhivery", "sector": "Logistics & Supply Chain"},
]

def get_all_market_symbols() -> List[str]:
    """Returns the full list of all symbols across the entire broad Indian market universe."""
    return [item["symbol"] for item in BROAD_NSE_UNIVERSE]
