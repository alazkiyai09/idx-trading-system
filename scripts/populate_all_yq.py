import sys
import os
import time

try:
    from yahooquery import Screener, Ticker
except ImportError:
    print("pip install yahooquery required")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.data.database import DatabaseManager

def fetch_and_populate():
    print("Fetching top Indonesian Equities via YahooQuery...")
    symbols = set()
    
    # Predefined list of 150+ highly liquid mid/large caps to supplement our 85.
    fallback_symbols = [
        "AMRT", "ASII", "BBCA", "BBNI", "BBRI", "BMRI", "CPIN", "GOTO", "ICBP", "INDF",
        "KLBF", "TLKM", "UNTR", "UNVR", "ADRO", "AKRA", "ANTM", "ARTOS", "BBTN", "BRPT",
        "BUKA", "EMTK", "ESSA", "EXCL", "HRUM", "INCO", "INKP", "INTP", "ISAT", "ITMG",
        "JPFA", "MDKA", "MEDC", "MIKA", "MNCN", "MYOR", "PGAS", "PTBA", "PTPP", "SIDO",
        "SMGR", "SMRA", "TBIG", "TINS", "TOWR", "TPIA", "WIKA", "ACES", "ADMR", "AGRO",
        "AMMN", "APLN", "ARTO", "BBYB", "BFIN", "BRIS", "BSDE", "CHEM", "CLEO", "CMRY",
        "CTRA", "DOID", "DSNG", "ELSA", "ENRG", "ERAA", "GGRM", "GJTL", "HEAL", "HMSP",
        "INDY", "JSMR", "LPKR", "LPPF", "MAPA", "MAPI", "MTEL", "PNBN", "PWON", "SCMA",
        "SMCB", "SRTG", "SSMS", "TAPG", "TKIM", "WTON", "AUTO", "BBHI", "BIRD", "BMHS",
        "BNBA", "BNGA", "BOBA", "BSML", "CINT", "CMNP", "DMMX", "DRMA", "FILM", "FREN",
        "GOOD", "HAIS", "HOKI", "IMJS", "IPCC", "IPPE", "IRRA", "KAEF", "KINO", "KPIG",
        "LCKM", "LINK", "LPCK", "LSIP", "MARK", "MBAP", "MCOL", "MFIN", "MLBI", "MLPL",
        "MPAZ", "MPMX", "MTDL", "NELY", "NISP", "OMED", "PANI", "PEVE", "PPRE", "PRDA",
        "RMKE", "ROTI", "SAME", "SILO", "SMMT", "SPMA", "SSIA", "STAA", "TOTO", "TRIN",
        "TRJA", "TSPC", "WIFI", "WOOD", "AALI", "ABBA", "AGII", "AISA", "ALDO", "AMAR",
        "BACA", "BANK", "BAPA", "BATA", "BGTG", "BIMA", "BINA", "BIPP", "BKDP", "BKSW",
        "BMAS", "BNII", "BNLI", "BPTR", "BSIM", "BTEK", "BTPS", "BVIC", "CARS", "CASS",
        "CBMF", "CENT", "CFIN", "CPRO", "CSMI", "DART", "DEFI", "DEWA", "DILD", "DIVA",
        "DSSA", "DUTI", "DVLA", "EAST", "EKAD", "ELTY", "EMDE", "EPMT", "ERTX", "FAST",
        "FASW", "FISH", "FMPB", "FORZ", "GAMA", "GDYR"
    ]
    
    if len(symbols) < 50:
        symbols.update([f"{s}.JK" for s in fallback_symbols])
        
    print(f"Total symbols to process: {len(symbols)}")
    
    db = DatabaseManager()
    session = db.get_session()
    from core.data.database import StockMetadata
    
    # Query existing symbols to avoid overwriting good names from TradingView
    existing = session.query(StockMetadata.symbol).all()
    existing_symbols = {e[0] for e in existing}
    
    batch_size = 50
    symbol_list = list(symbols)
    
    new_metadata = []
    
    for i in range(0, len(symbol_list), batch_size):
        batch = symbol_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(symbol_list)//batch_size)+1}...")
        
        # We can query all at once with YahooQuery Ticker
        tickers = Ticker(batch)
        try:
            profile_data = tickers.asset_profile
            price_data = tickers.price
            
            for sym in batch:
                clean_sym = sym.replace('.JK', '')
                if clean_sym in existing_symbols:
                    continue # Skip existing
                    
                prof = profile_data.get(sym, {})
                prc = price_data.get(sym, {})
                
                if isinstance(prof, str) or isinstance(prc, str):
                    continue # Error message returned
                if not prof or not prc:
                    continue
                    
                sector = prof.get('sector', 'Unknown')
                sub_sector = prof.get('industry', 'Unknown')
                name = prc.get('longName', prc.get('shortName', clean_sym))
                mcap = prc.get('marketCap', 0)
                
                meta = {
                    "symbol": clean_sym,
                    "name": name,
                    "sector": sector,
                    "sub_sector": sub_sector,
                    "market_cap": mcap,
                    "is_lq45": False,
                    "is_idx30": False
                }
                new_metadata.append(meta)
                print(f"  Added: {clean_sym} - {name} ({sector})")
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            
    if new_metadata:
        db.save_stock_metadata(new_metadata)
        print(f"Successfully added {len(new_metadata)} new stocks to database.")
    else:
        print("No new valid stocks added.")
        
if __name__ == "__main__":
    fetch_and_populate()
