import requests
from bs4 import BeautifulSoup
import pandas as pd

# List to store all products
products = []

# Base URL template
base_url = "http://books.toscrape.com/catalogue/page-{}.html"

# Loop through pages (1 to 50)
for page in range(1, 51):
    if page == 1:
        url = "http://books.toscrape.com/"
    else:
        url = base_url.format(page)
    
    res = requests.get(url)
    
    # Check if page exists
    if res.status_code != 200:
        print(f"Page {page} not found (status {res.status_code}). Stopping.")
        break
    
    # Fix encoding for pound symbol
    res.encoding = 'utf-8'
    
    # Parse HTML
    soup = BeautifulSoup(res.text, "html.parser")
    
    # Find all products on the page
    items = soup.select(".product_pod")
    if not items:
        print(f"No products found on page {page}. Stopping.")
        break
    
    # Extract product info
    for item in items:
        name = item.select_one("h3 a")["title"]
        price = item.select_one(".price_color").text.strip()
        products.append({
            "Product": name,
            "Price": price,
            "Discount": "0%"  # default since website has no discount info
        })
    
    print(f"Page {page}: Found {len(items)} products")

# Save all data to CSV
df = pd.DataFrame(products)
df.to_csv("ecommerce_purchases_1000s.csv", index=False, encoding="utf-8-sig")

print(f"\nFinished! Collected {len(df)} products and saved to 'ecommerce_purchases_1000s.csv'")
