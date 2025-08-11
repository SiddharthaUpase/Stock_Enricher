import pandas as pd
import numpy as np
import logging
import finnhub
import time
import re
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioData:
    """Class to represent and validate portfolio data"""
    
    def __init__(self, name: str = "", symbol: str = "", price: float = 0.0, 
                 shares: int = 0, market_value: float = 0.0):
        self.name = name.strip() if name else ""
        self.symbol = symbol.strip().upper() if symbol else ""
        self.price = price
        self.shares = shares
        self.market_value = market_value
        self.enriched = False  # Track if this entry was enriched
        
    def is_valid(self) -> bool:
        """Check if the portfolio entry has minimum required data"""
        return bool(self.name or self.symbol) and self.price >= 0 and self.shares >= 0
    
    def needs_symbol_lookup(self) -> bool:
        """Check if this entry needs symbol lookup"""
        return bool(self.name and not self.symbol)
    
    def needs_name_lookup(self) -> bool:
        """Check if this entry needs name lookup"""
        return bool(self.symbol and not self.name)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame compatibility"""
        return {
            "Name": self.name,
            "Symbol": self.symbol,
            "Price": self.price,
            "# of Shares": self.shares,
            "Market Value": self.market_value
        }

class FinnhubAPI:
    """Finnhub API client for symbol lookup using official library"""
    
    def __init__(self, api_key: str = ""):
        self.client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
        self.last_request_time = 0
        self.min_request_interval = 1  # 1 second between requests to avoid rate limits
        self.rate_limit_hit = False
        
    def _rate_limit(self):
        """Apply rate limiting between API calls"""
        # If we hit rate limit recently, wait longer
        if self.rate_limit_hit:
            logger.info("⏳ Waiting extra time due to recent rate limit...")
            time.sleep(5.0)  # Wait 5 seconds after rate limit
            self.rate_limit_hit = False
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def symbol_lookup(self, query: str) -> List[Dict]:
        """
        Look up symbols based on company name using Finnhub library
        
        Args:
            query: Company name or partial name
            
        Returns:
            List of matching symbols with metadata
        """
        if not query or len(query.strip()) < 2:
            return []
        
        # Clean query - remove common suffixes that might confuse search
        cleaned_query = self._clean_company_name(query)
        
        logger.info(f"Looking up symbol for: '{query}' (cleaned: '{cleaned_query}')")
        
        try:
            self._rate_limit()
            
            # Use the Finnhub library's symbol_lookup method
            response = self.client.symbol_lookup(cleaned_query)
            
            if response and 'result' in response:
                results = response['result']
                logger.info(f"Found {len(results)} results for '{query}'")
                return results
            else:
                logger.warning(f"No results found for '{query}'")
                return []
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "API limit reached" in error_msg:
                logger.warning(f"⏳ Rate limit hit for '{query}': {error_msg}")
                self.rate_limit_hit = True
                # Wait longer for rate limit recovery
                time.sleep(10)  # Wait 10 seconds for rate limit recovery
            else:
                logger.error(f"API request failed for '{query}': {error_msg}")
            return []
    
    def name_lookup(self, symbol: str) -> Optional[str]:
        """
        Look up company name based on symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Company name if found, None otherwise
        """
        if not symbol or len(symbol.strip()) < 1:
            return None
        
        symbol = symbol.strip().upper()
        logger.info(f"Looking up name for symbol: '{symbol}'")
        
        try:
            # Use symbol lookup with the symbol itself as query
            # This often returns the company associated with that symbol
            results = self.symbol_lookup(symbol)
            
            if results:
                # Look for exact symbol match
                for result in results:
                    result_symbol = result.get('symbol', '').upper()
                    # Remove exchange suffix for comparison (e.g., 'AAPL.US' -> 'AAPL')
                    clean_result_symbol = result_symbol.split('.')[0]
                    clean_input_symbol = symbol.split('.')[0]
                    
                    if clean_result_symbol == clean_input_symbol:
                        company_name = result.get('description', '')
                        if company_name:
                            logger.info(f"✓ Name lookup success: '{symbol}' → '{company_name}'")
                            return company_name
                
                # If no exact match, try the first result that contains the symbol
                for result in results:
                    result_symbol = result.get('symbol', '').upper()
                    if symbol in result_symbol or result_symbol.startswith(symbol):
                        company_name = result.get('description', '')
                        if company_name:
                            logger.info(f"✓ Partial name lookup success: '{symbol}' → '{company_name}'")
                            return company_name
            
            logger.warning(f"❌ Name lookup failed for symbol: '{symbol}'")
            return None
            
        except Exception as e:
            logger.error(f"Name lookup failed for '{symbol}': {str(e)}")
            return None
    
    def _clean_company_name(self, name: str) -> str:
        """Clean company name for better API search results"""
        if not name:
            return ""
        
        # Common company suffixes to remove for better matching
        suffixes_to_remove = [
            r'\s+Inc\.?$', r'\s+Corporation$', r'\s+Corp\.?$', r'\s+Company$', 
            r'\s+Co\.?$', r'\s+Ltd\.?$', r'\s+Limited$', r'\s+LLC$', 
            r'\s+L\.P\.?$', r'\s+LP$', r'\s+PLC$', r'\s+AG$', r'\s+SA$',
            r'\s+Class [A-Z]$', r'\s+\([^)]*\)$'  # Remove class designations and parentheses
        ]
        
        cleaned = name.strip()
        
        for suffix_pattern in suffixes_to_remove:
            cleaned = re.sub(suffix_pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def find_best_match(self, query: str) -> Optional[Dict]:
        """
        Find the best matching symbol for a company name using deterministic algorithm
        
        Returns:
            Best match dictionary with symbol info, or None if no good match
        """
        logger.info(f"🎯 Starting deterministic lookup for: '{query}'")
        
        # Step 1: Convert to Finnhub format
        finnhub_format = self._convert_to_finnhub_format(query)
        logger.info(f"📝 CSV Format: '{query}' → Finnhub Format: '{finnhub_format}'")
        
        # Step 2: Generate search variations
        variations = self._generate_search_variations(finnhub_format)
        logger.info(f"🔄 Generated {len(variations)} search variations: {variations}")
        
        # Step 3: Handle Class A/B/C companies specially
        best_match = None
        
        if 'class' in query.lower():
            logger.info("🎯 Class A/B/C company detected - searching for base company first")
            best_match = self._handle_class_company(query, finnhub_format)
        else:
            # Regular company - try variations normally
            best_match = self._search_variations(variations, finnhub_format)
        
        if best_match:
            logger.info(f"✅ SUCCESS: '{query}' → {best_match['symbol']} ({best_match.get('description', 'N/A')})")
            return best_match
        else:
            logger.error(f"❌ FAILED: No reliable match found for '{query}'")
            return None
    

    
    def _convert_to_finnhub_format(self, csv_name: str) -> str:
        """
        Convert CSV company name to Finnhub description format
        Examples:
        "Apple Inc." → "APPLE INC"
        "Microsoft Corporation" → "MICROSOFT CORP" 
        "Berkshire Hathaway Inc. Class B" → "BERKSHIRE HATHAWAY INC-CL B"
        """
        if not csv_name:
            return ""
        
        converted = csv_name.strip()
        
        # Step 1: Convert to uppercase
        converted = converted.upper()
        
        # Step 2: Handle Class A/B/C patterns
        # "Class A/B/C" → "-CL A/B/C"
        converted = re.sub(r'\s+CLASS\s+([ABC])', r'-CL \1', converted, flags=re.IGNORECASE)
        
        # Step 3: Normalize common suffixes
        # "Inc." → "INC"
        converted = re.sub(r'\bINC\.', 'INC', converted)
        # "Corp." → "CORP"
        converted = re.sub(r'\bCORP\.', 'CORP', converted)
        # "Corporation" → "CORP"
        converted = re.sub(r'\bCORPORATION\b', 'CORP', converted)
        # "Co." → "CO"
        converted = re.sub(r'\bCO\.', 'CO', converted)
        # "Company" → "CO"
        converted = re.sub(r'\bCOMPANY\b', 'CO', converted)
        # "Ltd." → "LTD"
        converted = re.sub(r'\bLTD\.', 'LTD', converted)
        # "Limited" → "LTD"
        converted = re.sub(r'\bLIMITED\b', 'LTD', converted)
        
        # Step 4: Clean up extra spaces
        converted = re.sub(r'\s+', ' ', converted).strip()
        
        return converted
    
    def _generate_search_variations(self, finnhub_format: str) -> List[str]:
        """Generate search variations for better matching"""
        variations = []
        
        # 1. Try the exact Finnhub format
        variations.append(finnhub_format)
        
        # 2. Try without common suffixes (often works better for search)
        without_suffix = finnhub_format
        suffixes = ['INC', 'CORP', 'CO', 'LTD', 'LLC']
        
        for suffix in suffixes:
            pattern = rf'\s+{suffix}$'
            if re.search(pattern, without_suffix, re.IGNORECASE):
                without_suffix = re.sub(pattern, '', without_suffix, flags=re.IGNORECASE).strip()
                variations.append(without_suffix)
                break  # Only remove one suffix
        
        # 3. Try title case version (mixed case often works better)
        title_case = ' '.join(word.capitalize() for word in without_suffix.split())
        
        if title_case != without_suffix and title_case != finnhub_format:
            variations.append(title_case)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for variation in variations:
            if variation not in seen:
                seen.add(variation)
                unique_variations.append(variation)
        
        return unique_variations
    
    def _search_variations(self, variations: List[str], finnhub_format: str) -> Optional[Dict]:
        """Search through variations to find the best match"""
        for i, variation in enumerate(variations):
            logger.info(f"🔍 Testing variation {i + 1}: '{variation}'")
            
            try:
                results = self.symbol_lookup(variation)
                
                if not results:
                    logger.info(f"   ❌ No results")
                    continue
                
                logger.info(f"   ✅ Found {len(results)} results:")
                for j, result in enumerate(results):
                    description = result.get('description', 'N/A')
                    symbol = result.get('symbol', 'N/A')
                    result_type = result.get('type', 'N/A')
                    logger.info(f"      {j + 1}. {symbol} - {description} (Type: {result_type})")
                    
                    # Check for exact description match
                    if description == finnhub_format:
                        logger.info(f"         🎯 EXACT MATCH with Finnhub format!")
                        return result
                
                # If we found results, use the first one as best match if no exact match
                if results:
                    best_match = results[0]
                    logger.info(f"      📌 Using first result as best match")
                    return best_match
                
            except Exception as e:
                logger.error(f"   ❌ API Error: {str(e)}")
                continue
            
            # Rate limiting between variations
            if i < len(variations) - 1:
                time.sleep(0.3)
        
        return None
    
    def _handle_class_company(self, csv_name: str, finnhub_format: str) -> Optional[Dict]:
        """Handle Class A/B/C companies by searching for base company first"""
        # Extract base company name (remove Class A/B/C)
        base_company = re.sub(r'\s+Class\s+[ABC]', '', csv_name, flags=re.IGNORECASE).strip()
        base_variations = self._generate_search_variations(self._convert_to_finnhub_format(base_company))
        
        logger.info(f"   📝 Base company: '{base_company}'")
        logger.info(f"   🔄 Base variations: {base_variations}")
        
        # Search for base company
        best_match = None
        for base_variation in base_variations:
            logger.info(f"   🔍 Searching base: '{base_variation}'")
            
            try:
                results = self.symbol_lookup(base_variation)
                
                if results:
                    best_match = results[0]
                    logger.info(f"   ✅ Found base company: {best_match['symbol']} - {best_match.get('description', 'N/A')}")
                    break
                else:
                    logger.info(f"   ❌ No results for base variation")
                
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"   ❌ Error: {str(e)}")
        
        if not best_match:
            return None
        
        # Handle Class A/B/C transformation if needed
        requested_class = None
        if 'class a' in csv_name.lower():
            requested_class = 'A'
        elif 'class b' in csv_name.lower():
            requested_class = 'B'
        elif 'class c' in csv_name.lower():
            requested_class = 'C'
        
        if requested_class:
            logger.info(f"🔄 Class {requested_class} detected, checking symbol transformation...")
            logger.info(f"   Found symbol: {best_match['symbol']}")
            
            # Transform symbol if needed
            target_symbol = best_match['symbol']
            needs_transformation = False
            
            # Handle different class transformations
            if requested_class == 'B' and best_match['symbol'].endswith('.A'):
                target_symbol = best_match['symbol'].replace('.A', '.B')
                needs_transformation = True
            elif requested_class == 'A' and best_match['symbol'].endswith('.B'):
                target_symbol = best_match['symbol'].replace('.B', '.A')
                needs_transformation = True
            elif requested_class == 'C':
                # For Class C, try common patterns (like GOOGL → GOOG for Alphabet)
                if best_match['symbol'] == 'GOOGL':
                    target_symbol = 'GOOG'
                    needs_transformation = True
                else:
                    # Try adding .C suffix for other companies
                    base_symbol = re.sub(r'\.[AB]$', '', best_match['symbol'])
                    target_symbol = base_symbol + '.C'
                    needs_transformation = True
            
            if needs_transformation:
                logger.info(f"   🔄 Transforming: {best_match['symbol']} → {target_symbol}")
                
                # Verify the transformed symbol exists
                time.sleep(0.5)
                logger.info(f"   🔍 Verifying transformed symbol...")
                
                try:
                    # Use company profile to verify the symbol exists
                    profile = self._get_company_profile(target_symbol)
                    if profile and profile.get('name'):
                        logger.info(f"   ✅ Verified: {target_symbol} - {profile['name']}")
                        # Create new result with transformed symbol
                        transformed_match = best_match.copy()
                        transformed_match['symbol'] = target_symbol
                        return transformed_match
                    else:
                        logger.info(f"   ❌ Transformed symbol not found, keeping original")
                        return best_match
                except Exception as e:
                    logger.error(f"   ❌ Verification error: {str(e)}")
                    return best_match
            else:
                logger.info(f"   ✅ No transformation needed")
                return best_match
        
        return best_match
    
    def _get_company_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile for symbol verification"""
        try:
            self._rate_limit()
            profile = self.client.company_profile2(symbol=symbol)
            return profile
        except Exception as e:
            logger.error(f"Profile lookup failed for '{symbol}': {str(e)}")
            return None

class SymbolEnricher:
    """Main class for enriching portfolio data with missing symbols"""
    
    def __init__(self, api_key: str = "d2aps11r01qgk9ug3ie0d2aps11r01qgk9ug3ieg"):
        self.csv_reader = CSVReader()
        self.finnhub = FinnhubAPI(api_key)
        
    def enrich_portfolio(self, portfolio_data: List[PortfolioData]) -> Tuple[List[PortfolioData], Dict]:
        """
        Enrich portfolio data by looking up missing symbols or names
        
        Returns:
            Tuple of (enriched_portfolio_data, enrichment_stats)
        """
        logger.info("Starting portfolio enrichment...")
        
        # Filter out entries with neither name nor symbol
        valid_entries = [item for item in portfolio_data if item.name or item.symbol]
        invalid_entries = [item for item in portfolio_data if not item.name and not item.symbol]
        
        if invalid_entries:
            logger.warning(f"Ignoring {len(invalid_entries)} entries with neither name nor symbol")
            for item in invalid_entries:
                logger.warning(f"  - Skipped entry: Price=${item.price}, Shares={item.shares}, Market Value=${item.market_value}")
        
        enrichment_stats = {
            "total_entries": len(portfolio_data),
            "valid_entries": len(valid_entries),
            "invalid_entries_skipped": len(invalid_entries),
            "attempted_symbol_lookups": 0,
            "attempted_name_lookups": 0,
            "successful_symbol_lookups": 0,
            "successful_name_lookups": 0,
            "failed_symbol_lookups": 0,
            "failed_name_lookups": 0,
            "already_complete": 0
        }
        
        for item in valid_entries:
            if item.needs_symbol_lookup():
                # Handle entries with name but no symbol
                enrichment_stats["attempted_symbol_lookups"] += 1
                
                # Use deterministic lookup algorithm
                match = self.finnhub.find_best_match(item.name)
                
                if match:
                    item.symbol = match['symbol']
                    item.enriched = True
                    enrichment_stats["successful_symbol_lookups"] += 1
                    logger.info(f"✓ Symbol lookup: '{item.name}' → {item.symbol}")
                else:
                    enrichment_stats["failed_symbol_lookups"] += 1
                    logger.warning(f"✗ Symbol lookup failed for '{item.name}'")
                        
            elif item.needs_name_lookup():
                # Handle entries with symbol but no name
                enrichment_stats["attempted_name_lookups"] += 1
                
                # Try name lookup
                company_name = self.finnhub.name_lookup(item.symbol)
                
                if company_name:
                    item.name = company_name
                    item.enriched = True
                    enrichment_stats["successful_name_lookups"] += 1
                    logger.info(f"✓ Name lookup: '{item.symbol}' → {item.name}")
                else:
                    enrichment_stats["failed_name_lookups"] += 1
                    logger.warning(f"✗ Name lookup failed for '{item.symbol}'")
                    
            elif item.symbol and item.name:
                enrichment_stats["already_complete"] += 1
        
        total_attempted = (enrichment_stats["attempted_symbol_lookups"] + 
                          enrichment_stats["attempted_name_lookups"])
        total_successful = (enrichment_stats["successful_symbol_lookups"] + 
                           enrichment_stats["successful_name_lookups"])
        
        success_rate = (total_successful / max(total_attempted, 1)) * 100
        
        logger.info(f"Enrichment completed - Overall success rate: {success_rate:.1f}%")
        logger.info(f"  - Symbol lookups: {enrichment_stats['successful_symbol_lookups']}/{enrichment_stats['attempted_symbol_lookups']}")
        logger.info(f"  - Name lookups: {enrichment_stats['successful_name_lookups']}/{enrichment_stats['attempted_name_lookups']}")
        
        return valid_entries, enrichment_stats
    


class CSVReader:
    """Robust CSV reader for portfolio data"""
    
    def __init__(self):
        self.required_columns = ["Name", "Symbol", "Price", "# of Shares", "Market Value"]
        self.alternative_column_names = {
            "Name": ["Company", "Company Name", "Security Name", "name"],
            "Symbol": ["Ticker", "Stock Symbol", "symbol", "ticker"],
            "Price": ["Share Price", "Unit Price", "price", "Current Price"],
            "# of Shares": ["Shares", "Quantity", "shares", "qty", "Share Count"],
            "Market Value": ["Total Value", "Position Value", "market_value", "value"]
        }
    
    def read_csv(self, file_path: str) -> Tuple[List[PortfolioData], Dict]:
        """
        Read and parse CSV file with robust error handling
        
        Returns:
            Tuple of (portfolio_data_list, metadata_dict)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Reading CSV file: {file_path}")
        
        try:
            # Try reading with different encodings
            df = self._read_with_encoding(file_path)
            
            # Normalize column names
            df = self._normalize_columns(df)
            
            # Validate required columns
            self._validate_columns(df)
            
            # Clean and validate data
            df = self._clean_data(df)
            
            # Convert to PortfolioData objects
            portfolio_data = self._convert_to_portfolio_data(df)
            
            # Generate metadata
            metadata = self._generate_metadata(portfolio_data, df)
            
            logger.info(f"Successfully loaded {len(portfolio_data)} portfolio entries")
            
            return portfolio_data, metadata
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise
    
    def _read_with_encoding(self, file_path: Path) -> pd.DataFrame:
        """Try reading CSV with different encodings"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully read CSV with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Unable to read CSV file with any supported encoding")
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format"""
        df_copy = df.copy()
        
        # Create mapping of current columns to standard columns
        column_mapping = {}
        
        for std_col, alternatives in self.alternative_column_names.items():
            # Add the standard column name to alternatives
            all_names = [std_col] + alternatives
            
            for col in df_copy.columns:
                if col.strip() in all_names or col.strip().lower() in [name.lower() for name in all_names]:
                    column_mapping[col] = std_col
                    break
        
        # Rename columns
        df_copy = df_copy.rename(columns=column_mapping)
        
        logger.info(f"Column mapping applied: {column_mapping}")
        
        return df_copy
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present"""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        if missing_columns:
            available_cols = list(df.columns)
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {available_cols}"
            )
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data types"""
        df_copy = df.copy()
        
        # Clean string columns
        string_columns = ["Name", "Symbol"]
        for col in string_columns:
            df_copy[col] = df_copy[col].astype(str).replace('nan', '').str.strip()
        
        # Clean numeric columns
        numeric_columns = ["Price", "# of Shares", "Market Value"]
        for col in numeric_columns:
            # Remove currency symbols and commas
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].astype(str).str.replace(r'[$,]', '', regex=True)
            
            # Convert to numeric, replacing errors with 0
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        
        # Ensure shares is integer
        df_copy["# of Shares"] = df_copy["# of Shares"].astype(int)
        
        logger.info("Data cleaning completed")
        
        return df_copy
    
    def _convert_to_portfolio_data(self, df: pd.DataFrame) -> List[PortfolioData]:
        """Convert DataFrame to list of PortfolioData objects"""
        portfolio_data = []
        
        for _, row in df.iterrows():
            try:
                data = PortfolioData(
                    name=row["Name"],
                    symbol=row["Symbol"],
                    price=float(row["Price"]),
                    shares=int(row["# of Shares"]),
                    market_value=float(row["Market Value"])
                )
                
                if data.is_valid():
                    portfolio_data.append(data)
                else:
                    logger.warning(f"Skipping invalid row: {row.to_dict()}")
                    
            except Exception as e:
                logger.warning(f"Error processing row {row.to_dict()}: {str(e)}")
                continue
        
        return portfolio_data
    
    def _generate_metadata(self, portfolio_data: List[PortfolioData], df: pd.DataFrame) -> Dict:
        """Generate metadata about the loaded data"""
        total_entries = len(portfolio_data)
        missing_symbols = sum(1 for item in portfolio_data if item.needs_symbol_lookup())
        missing_names = sum(1 for item in portfolio_data if item.needs_name_lookup())
        complete_entries = total_entries - missing_symbols - missing_names
        
        total_market_value = sum(item.market_value for item in portfolio_data)
        
        metadata = {
            "total_entries": total_entries,
            "missing_symbols": missing_symbols,
            "missing_names": missing_names,
            "complete_entries": complete_entries,
            "total_market_value": total_market_value,
            "original_rows": len(df),
            "invalid_rows_skipped": len(df) - total_entries
        }
        
        return metadata

def export_to_csv(portfolio_data: List[PortfolioData], filename: str = "enriched_portfolio.csv"):
    """Export enriched portfolio data to CSV (only entries with symbols)"""
    
    # Filter out entries without symbols
    entries_with_symbols = [item for item in portfolio_data if item.symbol]
    entries_without_symbols = [item for item in portfolio_data if not item.symbol]
    
    if entries_without_symbols:
        logger.warning(f"Excluding {len(entries_without_symbols)} entries without symbols from CSV:")
        for item in entries_without_symbols:
            logger.warning(f"  - {item.name}")
    
    if not entries_with_symbols:
        logger.error("No entries with symbols to export!")
        return None
    
    # Convert to DataFrame
    data_dicts = [item.to_dict() for item in entries_with_symbols]
    df = pd.DataFrame(data_dicts)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    logger.info(f"Enriched portfolio data exported to {filename}")
    logger.info(f"  - Exported entries: {len(entries_with_symbols)}")
    logger.info(f"  - Excluded entries: {len(entries_without_symbols)}")
    
    return filename

def main():
    """Main function demonstrating the complete enrichment workflow"""
    
    print("🚀 Stock Portfolio Symbol Enricher")
    print("=" * 50)
    
    # Initialize enricher
    enricher = SymbolEnricher()
    
    try:
        # Step 1: Read CSV data
        print("\n📖 Step 1: Reading CSV data...")
        portfolio_data, metadata = enricher.csv_reader.read_csv("dummy_portfolio.csv")
        
        print(f"✓ Loaded {metadata['total_entries']} entries")
        print(f"  - Missing symbols: {metadata['missing_symbols']}")
        print(f"  - Missing names: {metadata['missing_names']}")
        print(f"  - Complete entries: {metadata['complete_entries']}")
        
        # Step 2: Enrich missing symbols and names
        missing_data = metadata['missing_symbols'] + metadata['missing_names']
        if missing_data > 0:
            print(f"\n🔍 Step 2: Looking up {metadata['missing_symbols']} missing symbols and {metadata['missing_names']} missing names...")
            enriched_data, enrich_stats = enricher.enrich_portfolio(portfolio_data)
            
            print(f"✓ Enrichment completed")
            print(f"  - Valid entries processed: {enrich_stats['valid_entries']}")
            print(f"  - Invalid entries skipped: {enrich_stats['invalid_entries_skipped']}")
            print(f"  - Successful symbol lookups: {enrich_stats['successful_symbol_lookups']}")
            print(f"  - Failed symbol lookups: {enrich_stats['failed_symbol_lookups']}")
            print(f"  - Successful name lookups: {enrich_stats['successful_name_lookups']}")
            print(f"  - Failed name lookups: {enrich_stats['failed_name_lookups']}")
            
            total_attempted = enrich_stats['attempted_symbol_lookups'] + enrich_stats['attempted_name_lookups']
            total_successful = enrich_stats['successful_symbol_lookups'] + enrich_stats['successful_name_lookups']
            success_rate = (total_successful/max(total_attempted,1)*100) if total_attempted > 0 else 100
            print(f"  - Overall success rate: {success_rate:.1f}%")
        else:
            print("\n✓ Step 2: No missing data to lookup")
            enriched_data = portfolio_data
            enrich_stats = {"successful_symbol_lookups": 0, "failed_symbol_lookups": 0, "successful_name_lookups": 0, "failed_name_lookups": 0}
        
        # Step 3: Export enriched data
        print(f"\n💾 Step 3: Exporting enriched data...")
        output_file = export_to_csv(enriched_data)
        if output_file:
            print(f"✓ Saved to {output_file}")
        else:
            print("❌ No data to export (no entries with symbols)")
        
        # Step 4: Show results summary
        print(f"\n📊 Final Results Summary:")
        print(f"=" * 30)
        
        for i, item in enumerate(enriched_data):
            symbol_status = "✓" if item.symbol else "❌"
            name_status = "✓" if item.name else "❌"
            enriched_mark = "🆕" if item.enriched else "  "
            
            print(f"{enriched_mark} {i+1:2d}. {item.name[:23]:23} | {item.symbol:6} | ${item.price:7.2f} | {item.shares:3} shares | ${item.market_value:8.2f} | N:{name_status} S:{symbol_status}")
        
        # Summary stats
        total_enriched = sum(1 for item in enriched_data if item.enriched)
        still_missing = sum(1 for item in enriched_data if item.needs_symbol_lookup())
        
        print(f"\n🎯 Enrichment Summary:")
        print(f"  - Total entries: {len(enriched_data)}")
        print(f"  - Successfully enriched: {total_enriched}")
        print(f"  - Still missing symbols: {still_missing}")
        print(f"  - Completion rate: {((len(enriched_data) - still_missing)/len(enriched_data)*100):.1f}%")
          
    except Exception as e:
        logger.error(f"Enrichment process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 