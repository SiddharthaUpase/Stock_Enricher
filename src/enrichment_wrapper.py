#!/usr/bin/env python3
"""
Wrapper around existing enrichment logic to add progress tracking
for Flask API integration
"""

import logging
import io
from typing import Callable, Optional
from src.stock_enricher import SymbolEnricher, CSVReader, export_to_csv

logger = logging.getLogger(__name__)

class EnrichmentWrapper:
    """Wrapper that adds progress tracking to existing enrichment logic"""
    
    def __init__(self):
        self.csv_reader = CSVReader()
        self.enricher = SymbolEnricher()
    
    def enrich_csv_with_progress(self, 
                                csv_file_path: str, 
                                task_id: str,
                                progress_callback: Callable[[str, dict], None]) -> str:
        """
        Enrich CSV file with progress tracking
        
        Args:
            csv_file_path: Path to the CSV file to process
            task_id: Unique task identifier
            progress_callback: Function to call with progress updates
            
        Returns:
            String containing the enriched CSV content
        """
        try:
            # Step 1: Read and parse CSV
            progress_callback(task_id, {
                "message": "Reading CSV file...",
                "percentage": 5
            })
            
            portfolio_data, metadata = self.csv_reader.read_csv(csv_file_path)
            total_companies = len(portfolio_data)
            
            logger.info(f"Task {task_id}: Loaded {total_companies} companies from CSV")
            
            progress_callback(task_id, {
                "total": total_companies,
                "processed": 0,
                "percentage": 10,
                "message": f"Loaded {total_companies} companies. Starting enrichment...",
                "successful": 0,
                "failed": 0
            })
            
            # Step 2: Enrich both symbols and names with progress tracking
            successful_lookups = 0
            failed_lookups = 0
            failed_symbol_lookups = []  # Track failed company names
            failed_name_lookups = []    # Track failed symbols
            
            for i, item in enumerate(portfolio_data):
                progress_percentage = 10 + (i / total_companies) * 80  # 10% to 90%
                
                if item.needs_symbol_lookup():
                    # Handle entries with name but no symbol
                    progress_callback(task_id, {
                        "processed": i,
                        "total": total_companies,
                        "percentage": progress_percentage,
                        "current_company": item.name,
                        "message": f"Looking up symbol for {item.name}...",
                        "successful": successful_lookups,
                        "failed": failed_lookups
                    })
                    
                    # Use deterministic lookup algorithm
                    match = self.enricher.finnhub.find_best_match(item.name)
                    
                    if match:
                        item.symbol = match['symbol']
                        item.enriched = True
                        successful_lookups += 1
                        logger.info(f"Task {task_id}: ✓ Symbol lookup: '{item.name}' → {item.symbol}")
                    else:
                        failed_lookups += 1
                        failed_symbol_lookups.append(item.name)
                        logger.warning(f"Task {task_id}: ✗ Symbol lookup failed for '{item.name}'")
                            
                elif item.needs_name_lookup():
                    # Handle entries with symbol but no name
                    progress_callback(task_id, {
                        "processed": i,
                        "total": total_companies,
                        "percentage": progress_percentage,
                        "current_company": item.symbol,
                        "message": f"Looking up name for {item.symbol}...",
                        "successful": successful_lookups,
                        "failed": failed_lookups
                    })
                    
                    # Try name lookup
                    company_name = self.enricher.finnhub.name_lookup(item.symbol)
                    
                    if company_name:
                        item.name = company_name
                        item.enriched = True
                        successful_lookups += 1
                        logger.info(f"Task {task_id}: ✓ Name lookup: '{item.symbol}' → {item.name}")
                    else:
                        failed_lookups += 1
                        failed_name_lookups.append(item.symbol)
                        logger.warning(f"Task {task_id}: ✗ Name lookup failed for '{item.symbol}'")
                        
                else:
                    # Entry already has both name and symbol, or skip if invalid
                    progress_callback(task_id, {
                        "processed": i,
                        "total": total_companies,
                        "percentage": progress_percentage,
                        "current_company": item.name or item.symbol,
                        "message": f"Skipping {item.name or item.symbol} (already complete)",
                        "successful": successful_lookups,
                        "failed": failed_lookups
                    })
            
            # Step 3: Export to CSV
            progress_callback(task_id, {
                "processed": total_companies,
                "total": total_companies,
                "percentage": 95,
                "message": "Generating enriched CSV...",
                "successful": successful_lookups,
                "failed": failed_lookups
            })
            
            # Filter out entries without symbols and convert to CSV string
            entries_with_symbols = [item for item in portfolio_data if item.symbol]
            entries_without_symbols = [item for item in portfolio_data if not item.symbol]
            
            if not entries_with_symbols:
                raise ValueError("No entries with symbols found after enrichment")
            
            # Convert to CSV string using StringIO
            csv_string = self._convert_to_csv_string(entries_with_symbols)
            
            # Final progress update
            total_attempted = len([item for item in portfolio_data if item.needs_symbol_lookup() or item.needs_name_lookup()])
            success_rate = (successful_lookups / max(total_attempted, 1)) * 100
            
            progress_callback(task_id, {
                "processed": total_companies,
                "total": total_companies,
                "percentage": 100,
                "message": f"Enrichment completed! Success rate: {success_rate:.1f}%",
                "successful": successful_lookups,
                "failed": failed_lookups,
                "exported_entries": len(entries_with_symbols),
                "excluded_entries": len(entries_without_symbols)
            })
            
            # Print failure analysis at the end
            logger.info(f"Task {task_id}: Enrichment completed successfully")
            logger.info(f"Task {task_id}: Exported {len(entries_with_symbols)} entries, excluded {len(entries_without_symbols)}")
            
            # Print detailed failure analysis
            if failed_symbol_lookups or failed_name_lookups:
                logger.info(f"\n{'='*80}")
                logger.info(f"Task {task_id}: FAILURE ANALYSIS")
                logger.info(f"{'='*80}")
                
                if failed_symbol_lookups:
                    logger.info(f"\n❌ FAILED SYMBOL LOOKUPS ({len(failed_symbol_lookups)} companies):")
                    for i, company_name in enumerate(failed_symbol_lookups, 1):
                        logger.info(f"   {i:2d}. \"{company_name}\"")
                    
                    logger.info(f"\n🔍 ANALYSIS OF FAILED SYMBOL LOOKUPS:")
                    logger.info(f"   - These company names could not be matched to stock symbols")
                    logger.info(f"   - Possible reasons: Typos, non-public companies, delisted stocks, or API limitations")
                    logger.info(f"   - Consider manual verification or alternative data sources")
                
                if failed_name_lookups:
                    logger.info(f"\n❌ FAILED NAME LOOKUPS ({len(failed_name_lookups)} symbols):")
                    for i, symbol in enumerate(failed_name_lookups, 1):
                        logger.info(f"   {i:2d}. \"{symbol}\"")
                    
                    logger.info(f"\n🔍 ANALYSIS OF FAILED NAME LOOKUPS:")
                    logger.info(f"   - These symbols could not be matched to company names")
                    logger.info(f"   - Possible reasons: Invalid symbols, delisted stocks, or API limitations")
                    logger.info(f"   - Consider manual verification or symbol validation")
                
                logger.info(f"\n📊 FAILURE STATISTICS:")
                logger.info(f"   - Total failed lookups: {len(failed_symbol_lookups) + len(failed_name_lookups)}")
                logger.info(f"   - Failed symbol lookups: {len(failed_symbol_lookups)}")
                logger.info(f"   - Failed name lookups: {len(failed_name_lookups)}")
                logger.info(f"   - Success rate: {success_rate:.1f}%")
                logger.info(f"{'='*80}")
            else:
                logger.info(f"\n🎉 Task {task_id}: Perfect success rate - no failed lookups!")
            
            # Return both CSV string and enrichment metadata
            enrichment_metadata = {
                "total_processed": total_companies,
                "successful_lookups": successful_lookups,
                "failed_lookups": failed_lookups,
                "success_rate": success_rate,
                "exported_entries": len(entries_with_symbols),
                "excluded_entries": len(entries_without_symbols),
                "failed_symbol_lookups": failed_symbol_lookups,
                "failed_name_lookups": failed_name_lookups
            }
            
            return csv_string, enrichment_metadata
            
        except Exception as e:
            logger.error(f"Task {task_id}: Enrichment failed: {str(e)}")
            progress_callback(task_id, {
                "message": f"Enrichment failed: {str(e)}",
                "percentage": 0
            })
            raise
    
    def _convert_to_csv_string(self, portfolio_data) -> str:
        """Convert portfolio data to CSV string"""
        import pandas as pd
        
        # Convert to DataFrame
        data_dicts = [item.to_dict() for item in portfolio_data]
        df = pd.DataFrame(data_dicts)
        
        # Convert to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        csv_buffer.close()
        
        return csv_string
    
    def validate_csv_format(self, csv_file_path: str) -> dict:
        """
        Validate CSV format and return basic info
        
        Returns:
            Dictionary with validation results
        """
        try:
            portfolio_data, metadata = self.csv_reader.read_csv(csv_file_path)
            
            return {
                "valid": True,
                "total_entries": metadata["total_entries"],
                "missing_symbols": metadata["missing_symbols"],
                "missing_names": metadata["missing_names"],
                "complete_entries": metadata["complete_entries"]
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            } 