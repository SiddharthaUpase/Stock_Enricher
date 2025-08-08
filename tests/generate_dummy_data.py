import csv
import random
from decimal import Decimal, ROUND_HALF_UP

def generate_dummy_portfolio_data():
    """Generate dummy portfolio data with company names but missing ticker symbols"""
    
    # Large, realistic dataset with diverse companies (60+ companies)
    companies_data = [
        # Major Tech companies (Easy cases)
        {"name": "Apple Inc.", "symbol": "", "price_range": (150, 200)},
        {"name": "Microsoft Corporation", "symbol": "", "price_range": (300, 350)},
        {"name": "Alphabet Inc. Class A", "symbol": "", "price_range": (130, 150)},
        {"name": "Amazon.com Inc.", "symbol": "", "price_range": (120, 140)},
        {"name": "Meta Platforms Inc.", "symbol": "", "price_range": (250, 300)},
        {"name": "Tesla Inc.", "symbol": "", "price_range": (200, 280)},
        {"name": "NVIDIA Corporation", "symbol": "", "price_range": (400, 500)},
        {"name": "Netflix Inc.", "symbol": "", "price_range": (350, 450)},
        {"name": "Adobe Inc.", "symbol": "", "price_range": (400, 500)},
        {"name": "Salesforce Inc.", "symbol": "", "price_range": (200, 250)},
        {"name": "Oracle Corporation", "symbol": "", "price_range": (80, 120)},
        {"name": "Intel Corporation", "symbol": "", "price_range": (40, 60)},
        {"name": "Advanced Micro Devices Inc.", "symbol": "", "price_range": (100, 150)},
        {"name": "Cisco Systems Inc.", "symbol": "", "price_range": (45, 55)},
        {"name": "International Business Machines Corporation", "symbol": "", "price_range": (120, 160)},
        
        # Financial Services (Mixed difficulty)
        {"name": "Berkshire Hathaway Inc. Class B", "symbol": "", "price_range": (350, 380)},
        {"name": "JPMorgan Chase & Co.", "symbol": "", "price_range": (140, 170)},
        {"name": "Bank of America Corporation", "symbol": "", "price_range": (30, 40)},
        {"name": "Wells Fargo & Company", "symbol": "", "price_range": (40, 50)},
        {"name": "Goldman Sachs Group Inc.", "symbol": "", "price_range": (300, 400)},
        {"name": "Morgan Stanley", "symbol": "", "price_range": (80, 120)},
        {"name": "Citigroup Inc.", "symbol": "", "price_range": (50, 70)},
        {"name": "American Express Company", "symbol": "", "price_range": (150, 200)},
        {"name": "The Charles Schwab Corporation", "symbol": "", "price_range": (60, 80)},
        {"name": "BlackRock Inc.", "symbol": "", "price_range": (700, 900)},
        
        # Healthcare & Pharmaceuticals
        {"name": "Johnson & Johnson", "symbol": "", "price_range": (150, 170)},
        {"name": "Pfizer Inc.", "symbol": "", "price_range": (25, 35)},
        {"name": "UnitedHealth Group Incorporated", "symbol": "", "price_range": (450, 550)},
        {"name": "Merck & Co. Inc.", "symbol": "", "price_range": (90, 120)},
        {"name": "AbbVie Inc.", "symbol": "", "price_range": (140, 180)},
        {"name": "Eli Lilly and Company", "symbol": "", "price_range": (400, 600)},
        {"name": "Bristol-Myers Squibb Company", "symbol": "", "price_range": (50, 70)},
        {"name": "Moderna Inc.", "symbol": "", "price_range": (80, 120)},
        {"name": "Gilead Sciences Inc.", "symbol": "", "price_range": (70, 90)},
        
        # Consumer Goods & Retail
        {"name": "The Coca-Cola Company", "symbol": "", "price_range": (55, 65)},
        {"name": "Procter & Gamble Company", "symbol": "", "price_range": (140, 160)},
        {"name": "Walmart Inc.", "symbol": "", "price_range": (50, 60)},
        {"name": "The Home Depot Inc.", "symbol": "", "price_range": (300, 350)},
        {"name": "McDonald's Corporation", "symbol": "", "price_range": (250, 300)},
        {"name": "Nike Inc.", "symbol": "", "price_range": (90, 130)},
        {"name": "Starbucks Corporation", "symbol": "", "price_range": (80, 110)},
        {"name": "Target Corporation", "symbol": "", "price_range": (120, 160)},
        {"name": "Costco Wholesale Corporation", "symbol": "", "price_range": (700, 900)},
        {"name": "PepsiCo Inc.", "symbol": "", "price_range": (160, 180)},
        
        # Industrial & Energy
        {"name": "General Electric Company", "symbol": "", "price_range": (80, 120)},
        {"name": "Caterpillar Inc.", "symbol": "", "price_range": (200, 300)},
        {"name": "Exxon Mobil Corporation", "symbol": "", "price_range": (90, 130)},
        {"name": "Chevron Corporation", "symbol": "", "price_range": (140, 180)},
        {"name": "The Boeing Company", "symbol": "", "price_range": (180, 250)},
        {"name": "3M Company", "symbol": "", "price_range": (90, 120)},
        {"name": "Honeywell International Inc.", "symbol": "", "price_range": (180, 220)},
        {"name": "Lockheed Martin Corporation", "symbol": "", "price_range": (400, 500)},
        {"name": "Raytheon Technologies Corporation", "symbol": "", "price_range": (80, 110)},
        
        # Telecommunications & Media
        {"name": "Verizon Communications Inc.", "symbol": "", "price_range": (35, 45)},
        {"name": "AT&T Inc.", "symbol": "", "price_range": (18, 25)},
        {"name": "Comcast Corporation", "symbol": "", "price_range": (40, 50)},
        {"name": "The Walt Disney Company", "symbol": "", "price_range": (80, 120)},
        {"name": "Charter Communications Inc.", "symbol": "", "price_range": (300, 400)},
        
        # Challenging/Edge Cases (These might be harder for APIs)
        {"name": "Berkshire Hathaway Inc. Class A", "symbol": "", "price_range": (400000, 500000)},  # Very expensive stock
        {"name": "Texas Instruments Incorporated", "symbol": "", "price_range": (160, 200)},  # Long name
        {"name": "American International Group Inc.", "symbol": "", "price_range": (50, 70)},  # AIG
        {"name": "Laboratory Corporation of America Holdings", "symbol": "", "price_range": (200, 250)},  # Very long name
        {"name": "Kimberly-Clark Corporation", "symbol": "", "price_range": (120, 150)},  # Hyphenated name
        {"name": "McCormick & Company Incorporated", "symbol": "", "price_range": (70, 90)},  # Ampersand
        {"name": "The Travelers Companies Inc.", "symbol": "", "price_range": (160, 200)},  # "The" prefix
        {"name": "AutoZone Inc.", "symbol": "", "price_range": (2500, 3000)},  # One word, expensive
        {"name": "O'Reilly Automotive Inc.", "symbol": "", "price_range": (800, 1000)},  # Apostrophe
        {"name": "S&P Global Inc.", "symbol": "", "price_range": (350, 450)},  # Special characters
        
        # International/Challenging
        {"name": "Taiwan Semiconductor Manufacturing Company Limited", "symbol": "", "price_range": (80, 120)},  # Very long
        {"name": "ASML Holding N.V.", "symbol": "", "price_range": (600, 800)},  # Dutch company
        {"name": "Shopify Inc.", "symbol": "", "price_range": (50, 80)},  # Canadian
        {"name": "Sea Limited", "symbol": "", "price_range": (40, 80)},  # Simple but might be tricky
    ]
    
    portfolio_data = []
    
    for company in companies_data:
        # Generate random price within range
        price = round(random.uniform(company["price_range"][0], company["price_range"][1]), 2)
        
        # Generate random number of shares (1-50 for variety)
        shares = random.randint(1, 50)
        
        # Calculate market value
        market_value = round(price * shares, 2)
        
        portfolio_data.append({
            "Name": company["name"],
            "Symbol": company["symbol"],
            "Price": price,
            "# of Shares": shares,
            "Market Value": market_value
        })
    
    return portfolio_data

def save_to_csv(data, filename="dummy_portfolio.csv"):
    """Save portfolio data to CSV file"""
    
    fieldnames = ["Name", "Symbol", "Price", "# of Shares", "Market Value"]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Generated dummy portfolio data saved to {filename}")
    print(f"Total entries: {len(data)}")
    
    # Print summary statistics
    missing_symbols = sum(1 for row in data if not row["Symbol"])
    missing_names = sum(1 for row in data if not row["Name"])
    complete_rows = sum(1 for row in data if row["Name"] and row["Symbol"])
    
    print(f"Entries missing symbols: {missing_symbols}")
    print(f"Entries missing names: {missing_names}")
    print(f"Complete entries: {complete_rows}")
    
    # Calculate total portfolio value
    total_value = sum(row["Market Value"] for row in data)
    print(f"Total portfolio value: ${total_value:,.2f}")

def preview_data(data, num_rows=10):
    """Preview the generated data"""
    print("\n--- Data Preview ---")
    print(f"{'Name':<30} {'Symbol':<8} {'Price':<8} {'Shares':<8} {'Market Value':<12}")
    print("-" * 75)
    
    for i, row in enumerate(data[:num_rows]):
        name = row["Name"][:28] + ".." if len(row["Name"]) > 30 else row["Name"]
        symbol = row["Symbol"] if row["Symbol"] else "MISSING"
        print(f"{name:<30} {symbol:<8} ${row['Price']:<7.2f} {row['# of Shares']:<8} ${row['Market Value']:<11.2f}")
    
    if len(data) > num_rows:
        print(f"... and {len(data) - num_rows} more entries")

if __name__ == "__main__":
    # Generate dummy data
    portfolio_data = generate_dummy_portfolio_data()
    
    # Preview the data
    preview_data(portfolio_data)
    
    # Save to CSV
    save_to_csv(portfolio_data) 