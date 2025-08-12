#!/usr/bin/env python3
"""
Test Enhanced Backend Algorithm
Tests if the enhanced search variations can find previously failed companies
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from stock_enricher import FinnhubAPI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test cases that previously failed
FAILED_COMPANIES = [
    {
        "name": "International Business Machines Corp",
        "expected_symbol": "IBM",
        "description": "IBM - Major tech company"
    },
    {
        "name": "Advanced Micro Devices Inc.",
        "expected_symbol": "AMD",
        "description": "AMD - Semiconductor company"
    },
    {
        "name": "Zoom Video Communications Inc.",
        "expected_symbol": "ZM",
        "description": "Zoom - Video conferencing company"
    }
]

def test_enhanced_algorithm():
    """Test the enhanced algorithm against previously failed companies"""
    
    print("🧪 TESTING ENHANCED BACKEND ALGORITHM")
    print("=" * 80)
    print("Testing companies that previously failed with the old algorithm")
    print("=" * 80)
    
    # Initialize the API client
    finnhub_api = FinnhubAPI()
    
    results = []
    
    for i, company in enumerate(FAILED_COMPANIES, 1):
        print(f"\n{'='*60}")
        print(f"🧪 TEST {i}/3: {company['name']}")
        print(f"   Expected Symbol: {company['expected_symbol']}")
        print(f"   Description: {company['description']}")
        print(f"{'='*60}")
        
        # Test the enhanced algorithm
        try:
            match = finnhub_api.find_best_match(company['name'])
            
            if match:
                found_symbol = match.get('symbol', 'N/A')
                found_description = match.get('description', 'N/A')
                
                if found_symbol == company['expected_symbol']:
                    print(f"✅ SUCCESS: Found correct symbol!")
                    print(f"   Symbol: {found_symbol}")
                    print(f"   Description: {found_description}")
                    results.append({"company": company['name'], "success": True, "symbol": found_symbol})
                else:
                    print(f"⚠️  PARTIAL: Found different symbol")
                    print(f"   Found: {found_symbol} - {found_description}")
                    print(f"   Expected: {company['expected_symbol']}")
                    results.append({"company": company['name'], "success": False, "symbol": found_symbol})
            else:
                print(f"❌ FAILED: No match found")
                results.append({"company": company['name'], "success": False, "symbol": None})
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({"company": company['name'], "success": False, "symbol": None, "error": str(e)})
    
    # Summary
    print(f"\n\n📊 ENHANCED ALGORITHM TEST RESULTS")
    print("=" * 80)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful matches: {len(successful)}/{len(FAILED_COMPANIES)}")
    print(f"❌ Failed matches: {len(failed)}/{len(FAILED_COMPANIES)}")
    print(f"📈 Success rate: {(len(successful) / len(FAILED_COMPANIES) * 100):.1f}%")
    
    if successful:
        print(f"\n✅ SUCCESSFUL MATCHES:")
        for result in successful:
            print(f"   • \"{result['company']}\" → {result['symbol']}")
    
    if failed:
        print(f"\n❌ FAILED MATCHES:")
        for result in failed:
            error_info = f" (Error: {result.get('error', 'N/A')})" if result.get('error') else ""
            symbol_info = f" → {result['symbol']}" if result.get('symbol') else ""
            print(f"   • \"{result['company']}\"{symbol_info}{error_info}")
    
    print(f"\n🎯 IMPROVEMENT ANALYSIS:")
    if len(successful) == len(FAILED_COMPANIES):
        print("   🎉 Perfect! All previously failed companies now work!")
        print("   💡 Enhanced search variations solved the search issues")
        print("   🚀 Ready to deploy enhanced algorithm")
    elif len(successful) > 0:
        print(f"   📈 Improved from 0% to {(len(successful) / len(FAILED_COMPANIES) * 100):.1f}% success rate")
        print("   🔧 Some companies still need additional strategies")
        print("   💡 Consider further enhancements for remaining failures")
    else:
        print("   😞 No improvement detected")
        print("   🔧 Enhanced variations may need debugging")
    
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    test_enhanced_algorithm() 