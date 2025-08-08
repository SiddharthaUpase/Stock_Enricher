#!/usr/bin/env python3
"""
LLM-based company name variation generator for stock symbol lookup
"""

import os
import re
import json
import logging
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMNameGenerator:
    """Generate stock-friendly company name variations using OpenAI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def generate_stock_friendly_names(self, company_name: str) -> List[str]:
        """
        Generate 4 stock-friendly variations of a company name using LLM
        
        Args:
            company_name: Original company name
            
        Returns:
            List of stock-friendly name variations
        """
        if not company_name or not company_name.strip():
            return []
        
        logger.info(f"Generating LLM variations for: '{company_name}'")
        
        try:
            prompt = self._create_prompt(company_name)
            response = self._call_openai(prompt)
            variations = self._extract_json_from_response(response)
            
            # Filter out empty/invalid variations
            variations = [v.strip() for v in variations if v and v.strip()]
            
            logger.info(f"LLM generated {len(variations)} variations: {variations}")
            return variations
            
        except Exception as e:
            logger.error(f"LLM name generation failed for '{company_name}': {str(e)}")
            return []
    
    def _create_prompt(self, company_name: str) -> str:
        """Create the prompt for OpenAI"""
        return f"""Given the company name "{company_name}", generate exactly 4 stock-friendly variations that would work well for financial API symbol lookup.

Rules:
1. Remove unnecessary legal suffixes (Inc, Corp, etc.)
2. Use common trading names when different from legal name
3. Include short forms that traders might use
4. Consider ticker symbol patterns
5. Return variations from most likely to work to least likely

Example:
For "Alphabet Inc. Class A", return:
```json
["Google", "Alphabet", "GOOGL", "Alphabet Inc"]
```

For "Berkshire Hathaway Inc. Class B", return:
```json
["Berkshire Hathaway", "Berkshire", "BRK.B", "BRK"]
```

Now for "{company_name}", return ONLY the JSON array in code blocks with no additional text:

```json
["variation1", "variation2", "variation3", "variation4"]
```"""
    
    def _call_openai(self, prompt: str) -> str:
        """Make API call to OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[
                    {"role": "system", "content": "You are a financial data expert specializing in company name normalization for stock symbol lookup."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def _extract_json_from_response(self, response: str) -> List[str]:
        """Extract JSON array from LLM response using regex"""
        if not response:
            return []
        
        logger.info(f"Raw LLM response: {response}")
        
        # Try multiple patterns to extract JSON array
        patterns = [
            r'```json\s*(\[.*?\])\s*```',  # Code block with json
            r'```\s*(\[.*?\])\s*```',      # Code block without json marker
            r'(\[.*?\])',                   # Standalone JSON array
        ]
        
        json_str = None
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                json_str = matches[0]
                logger.info(f"Found JSON using pattern: {pattern}")
                logger.info(f"Extracted JSON string: {json_str}")
                break
        
        if not json_str:
            logger.warning(f"No JSON array found in LLM response: {response}")
            return []
        
        try:
            # Parse the JSON
            variations = json.loads(json_str)
            
            if not isinstance(variations, list):
                logger.warning(f"LLM returned non-list: {variations}")
                return []
            
            # Convert all items to strings and validate
            string_variations = []
            for item in variations:
                if isinstance(item, str) and item.strip():
                    string_variations.append(item.strip())
            
            logger.info(f"Successfully parsed {len(string_variations)} variations: {string_variations}")
            return string_variations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {json_str}, Error: {e}")
            return []

def test_llm_generator():
    """Test function for the LLM name generator"""
    generator = LLMNameGenerator()
    
    test_companies = [
        "Alphabet Inc. Class A",
        "Berkshire Hathaway Inc. Class B",
        "Meta Platforms Inc.",
        "The Coca-Cola Company"
    ]
    
    print("🤖 Testing LLM Name Generator")
    print("=" * 50)
    
    for company in test_companies:
        print(f"\n🏢 Testing: {company}")
        print("-" * 30)
        
        variations = generator.generate_stock_friendly_names(company)
        
        if variations:
            for i, variation in enumerate(variations, 1):
                print(f"  {i}. {variation}")
        else:
            print("  ❌ No variations generated")

if __name__ == "__main__":
    test_llm_generator() 