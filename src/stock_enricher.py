import pandas as pd
import numpy as np
import logging
import finnhub
import time
import re
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from src.llm_name_generator import LLMNameGenerator
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
        self.min_request_interval = 0.5  # 0.5 second between requests to avoid rate limits
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
        Find the best matching symbol for a company name
        
        Returns:
            Best match dictionary with symbol info, or None if no good match
        """
        results = self.symbol_lookup(query)
        
        if not results:
            logger.error(f"❌ LOOKUP FAILED for '{query}': No API results returned")
            return None
        
        # Score results based on relevance
        scored_results = []
        query_lower = query.lower()
        
        for result in results:
            score = self._calculate_match_score(query_lower, result)
            scored_results.append((score, result))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return best match if score is good enough
        best_score, best_match = scored_results[0]
        
        if best_score > 0.3:  # Minimum confidence threshold
            logger.info(f"✓ Best match for '{query}': {best_match['symbol']} ({best_match['description']}) - Score: {best_score:.2f}")
            return best_match
        else:
            # Detailed failure analysis
            self._log_detailed_failure_analysis(query, scored_results)
            return None
    
    def _log_detailed_failure_analysis(self, query: str, scored_results: List[Tuple[float, Dict]]):
        """Log detailed analysis of why a lookup failed"""
        logger.error(f"\n❌ DETAILED FAILURE ANALYSIS for '{query}':")
        logger.error(f"   Original query: '{query}'")
        logger.error(f"   Cleaned query: '{self._clean_company_name(query)}'")
        logger.error(f"   Number of API results: {len(scored_results)}")
        logger.error(f"   Confidence threshold: 0.3")
        
        if scored_results:
            logger.error(f"   TOP 3 RESULTS WITH SCORES:")
            for i, (score, result) in enumerate(scored_results[:3]):
                symbol = result.get('symbol', 'N/A')
                description = result.get('description', 'N/A')
                result_type = result.get('type', 'N/A')
                logger.error(f"     #{i+1}: {symbol:8} | {description:30} | Type: {result_type:15} | Score: {score:.3f}")
            
            # Show why the best match failed
            best_score, best_result = scored_results[0]
            logger.error(f"   BEST MATCH ANALYSIS:")
            logger.error(f"     Symbol: {best_result.get('symbol', 'N/A')}")
            logger.error(f"     Description: {best_result.get('description', 'N/A')}")
            logger.error(f"     Score: {best_score:.3f} (needed: 0.3)")
            logger.error(f"     Failure reason: Score below confidence threshold")
            
            # Show detailed scoring breakdown
            self._log_scoring_breakdown(query, best_result, best_score)
        else:
            logger.error(f"   No results returned from API")
    
    def _log_scoring_breakdown(self, query: str, result: Dict, total_score: float):
        """Log detailed breakdown of how the score was calculated"""
        query_lower = query.lower()
        description = result.get('description', '').lower()
        symbol = result.get('symbol', '').lower()
        
        logger.error(f"   SCORING BREAKDOWN:")
        logger.error(f"     Query: '{query_lower}'")
        logger.error(f"     Description: '{description}'")
        
        # Exact match analysis
        if query_lower == description:
            logger.error(f"     ✓ Exact match: +1.0")
        elif query_lower in description:
            logger.error(f"     ✓ Query in description: +0.8")
        elif description.startswith(query_lower):
            logger.error(f"     ✓ Description starts with query: +0.7")
        else:
            logger.error(f"     ✗ No exact/substring match: +0.0")
        
        # Word match analysis
        query_words = set(query_lower.split())
        description_words = set(description.split())
        matching_words = query_words & description_words
        
        if query_words:
            word_match_ratio = len(matching_words) / len(query_words)
            logger.error(f"     Word overlap: {len(matching_words)}/{len(query_words)} = {word_match_ratio:.2f} → +{word_match_ratio * 0.6:.3f}")
            logger.error(f"       Query words: {query_words}")
            logger.error(f"       Description words: {description_words}")
            logger.error(f"       Matching words: {matching_words}")
        
        # Acronym analysis
        acronym_score = self._calculate_acronym_score(query_lower, description, result.get('symbol', ''))
        if acronym_score > 0:
            logger.error(f"     ✓ Acronym match bonus: +{acronym_score:.3f}")
        else:
            logger.error(f"     ✗ No acronym match: +0.0")
        
        # Bonus points
        if result.get('symbol', '').endswith('.US') or '.' not in result.get('symbol', ''):
            logger.error(f"     ✓ US exchange bonus: +0.1")
        else:
            logger.error(f"     ✗ Non-US exchange: +0.0")
            
        if result.get('type', '').lower() == 'common stock':
            logger.error(f"     ✓ Common stock bonus: +0.1")
        else:
            logger.error(f"     ✗ Not common stock: +0.0")
        
        logger.error(f"     TOTAL SCORE: {total_score:.3f}")
        logger.error(f"")  # Empty line for readability
    
    def _calculate_match_score(self, query: str, result: Dict) -> float:
        """Calculate match score between query and result"""
        description = result.get('description', '').lower()
        symbol = result.get('symbol', '').lower()
        
        score = 0.0
        
        # Exact matches get highest score
        if query == description:
            score += 1.0
        elif query in description:
            score += 0.8
        elif description.startswith(query):
            score += 0.7
        
        # Check for word matches
        query_words = set(query.split())
        description_words = set(description.split())
        
        if query_words:
            word_match_ratio = len(query_words & description_words) / len(query_words)
            score += word_match_ratio * 0.6
        
        # NEW: Check for acronym matches
        acronym_score = self._calculate_acronym_score(query, description, symbol)
        score += acronym_score
        
        # Prefer US exchange symbols
        if result.get('symbol', '').endswith('.US') or '.' not in result.get('symbol', ''):
            score += 0.1
        
        # Prefer common stock
        if result.get('type', '').lower() == 'common stock':
            score += 0.1
        
        return score
    
    def _calculate_acronym_score(self, query: str, description: str, symbol: str) -> float:
        """Calculate additional score for acronym matches"""
        
        # Check if query matches symbol (case insensitive)
        if query.upper() == symbol.upper():
            # Check if the symbol could be an acronym of the description
            desc_words = description.split()
            if len(desc_words) >= 2:
                # Create acronym from first letters of description words
                potential_acronym = ''.join(word[0] for word in desc_words if word).upper()
                
                if query.upper() == potential_acronym:
                    return 0.5  # Strong acronym match
                
                # Also check if query matches major words (skip common words)
                major_words = [word for word in desc_words if len(word) > 2 and word.lower() not in ['inc', 'corp', 'company', 'ltd', 'the', 'and', 'of']]
                if len(major_words) >= 2:
                    major_acronym = ''.join(word[0] for word in major_words).upper()
                    if query.upper() == major_acronym:
                        return 0.4  # Good acronym match
        
        # Check if description words start with query letters
        if len(query) >= 2:
            desc_words = description.split()
            if len(desc_words) >= len(query):
                # Check if each letter of query matches first letter of description words
                matches = 0
                for i, letter in enumerate(query.lower()):
                    if i < len(desc_words) and desc_words[i].lower().startswith(letter):
                        matches += 1
                
                if matches == len(query) and matches >= 2:
                    return 0.3  # Partial acronym match
        
        return 0.0

class SymbolEnricher:
    """Main class for enriching portfolio data with missing symbols"""
    
    def __init__(self, api_key: str = "d2aps11r01qgk9ug3ie0d2aps11r01qgk9ug3ieg"):
        self.csv_reader = CSVReader()
        self.finnhub = FinnhubAPI(api_key)
        self.llm_generator = LLMNameGenerator()
        
    def enrich_portfolio(self, portfolio_data: List[PortfolioData]) -> Tuple[List[PortfolioData], Dict]:
        """
        Enrich portfolio data by looking up missing symbols
        
        Returns:
            Tuple of (enriched_portfolio_data, enrichment_stats)
        """
        logger.info("Starting portfolio enrichment...")
        
        enrichment_stats = {
            "total_entries": len(portfolio_data),
            "attempted_lookups": 0,
            "successful_lookups": 0,
            "failed_lookups": 0,
            "already_complete": 0
        }
        
        for item in portfolio_data:
            if item.needs_symbol_lookup():
                enrichment_stats["attempted_lookups"] += 1
                
                # Try direct lookup first
                match = self.finnhub.find_best_match(item.name)
                
                if match:
                    item.symbol = match['symbol']
                    item.enriched = True
                    enrichment_stats["successful_lookups"] += 1
                    logger.info(f"✓ Direct match: '{item.name}' → {item.symbol}")
                else:
                    # LLM fallback - try name variations
                    logger.info(f"🤖 Trying LLM fallback for '{item.name}'")
                    match = self._try_llm_fallback(item.name)
                    
                    if match:
                        item.symbol = match['symbol']
                        item.enriched = True
                        enrichment_stats["successful_lookups"] += 1
                        logger.info(f"✓ LLM fallback success: '{item.name}' → {item.symbol}")
                    else:
                        enrichment_stats["failed_lookups"] += 1
                        logger.warning(f"✗ All methods failed for '{item.name}'")
                    
            elif item.symbol and item.name:
                enrichment_stats["already_complete"] += 1
        
        success_rate = (enrichment_stats["successful_lookups"] / 
                       max(enrichment_stats["attempted_lookups"], 1)) * 100
        
        logger.info(f"Enrichment completed - Success rate: {success_rate:.1f}%")
        
        return portfolio_data, enrichment_stats
    
    def _try_llm_fallback(self, company_name: str) -> Optional[Dict]:
        """
        Try LLM-generated name variations as fallback
        
        Args:
            company_name: Original company name that failed direct lookup
            
        Returns:
            Match dictionary if found, None otherwise
        """
        try:
            # Generate variations using LLM
            variations = self.llm_generator.generate_stock_friendly_names(company_name)
            
            if not variations:
                logger.warning(f"LLM generated no variations for '{company_name}'")
                return None
            
            # Try each variation
            for i, variation in enumerate(variations, 1):
                logger.info(f"  Trying variation #{i}: '{variation}'")
                match = self.finnhub.find_best_match(variation)
                
                if match:
                    logger.info(f"  ✓ Success with variation #{i}: '{variation}' → {match['symbol']}")
                    return match
                else:
                    logger.info(f"  ❌ Variation #{i} failed: '{variation}'")
            
            logger.warning(f"All {len(variations)} LLM variations failed for '{company_name}'")
            return None
            
        except Exception as e:
            logger.error(f"LLM fallback failed for '{company_name}': {str(e)}")
            return None

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
        
        # Step 2: Enrich missing symbols
        if metadata['missing_symbols'] > 0:
            print(f"\n🔍 Step 2: Looking up {metadata['missing_symbols']} missing symbols...")
            enriched_data, enrich_stats = enricher.enrich_portfolio(portfolio_data)
            
            print(f"✓ Symbol lookup completed")
            print(f"  - Successful lookups: {enrich_stats['successful_lookups']}")
            print(f"  - Failed lookups: {enrich_stats['failed_lookups']}")
            print(f"  - Success rate: {(enrich_stats['successful_lookups']/max(enrich_stats['attempted_lookups'],1)*100):.1f}%")
        else:
            print("\n✓ Step 2: No missing symbols to lookup")
            enriched_data = portfolio_data
            enrich_stats = {"successful_lookups": 0, "failed_lookups": 0}
        
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