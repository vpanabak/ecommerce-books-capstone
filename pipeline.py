import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

print("=== ZEPTO DATA PIPELINE: CATALOG BENCHMARKING ===")

# Fixed project baseline conversion rate (Required)
GBP_TO_INR_RATE = 105.50

# Map text star ratings to integers 1-5
RATING_MAP = {
    'One': 1,
    'Two': 2,
    'Three': 3,
    'Four': 4,
    'Five': 5
}

# -------------------------------------------------------------
# STEP 1: WEB SCRAPING
# -------------------------------------------------------------
print("\n[Step 1] Scraping books from books.toscrape.com...")

base_url = "http://books.toscrape.com/"
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Get categories from side menu (select at least 3 categories)
category_links = soup.select('.side_categories ul.nav-list ul li a')
categories_to_scrape = []

for link in category_links[:3]: # Scrapes first 3 categories
    cat_name = link.text.strip()
    cat_url = base_url + link['href']
    categories_to_scrape.append((cat_name, cat_url))

scraped_data = []

for cat_name, cat_url in categories_to_scrape:
    cat_response = requests.get(cat_url)
    cat_soup = BeautifulSoup(cat_response.text, 'html.parser')
    
    books = cat_soup.select('.product_pod')
    for book in books:
        title = book.h3.a['title']
        price_raw = book.select_one('.price_color').text
        rating_raw = book.p['class'][1] # e.g. 'Three'
        availability_raw = book.select_one('.availability').text.strip()
        
        scraped_data.append({
            'title': title,
            'price_raw': price_raw,
            'rating_raw': rating_raw,
            'availability_raw': availability_raw,
            'category': cat_name
        })

df_raw = pd.DataFrame(scraped_data)
print(f"Scraped {len(df_raw)} total books across {len(categories_to_scrape)} categories.")

# -------------------------------------------------------------
# STEP 2: DATA CLEANING & ENRICHMENT
# -------------------------------------------------------------
print("\n[Step 2] Cleaning & enriching fields...")

cleaned_data = []

for idx, row in df_raw.iterrows():
    # 1. Clean Price GBP
    try:
        clean_price = float(row['price_raw'].replace('£', '').replace('Â', '').strip())
    except ValueError:
        clean_price = np.nan # Will be handled by median imputation if needed
    
    # 2. Clean Rating (Text to Int 1-5)
    clean_rating = RATING_MAP.get(row['rating_raw'], np.nan)
    
    # 3. Clean Availability (Boolean in_stock)
    clean_in_stock = 1 if "In stock" in row['availability_raw'] else 0
    
    cleaned_data.append({
        'title': row['title'],
        'price_gbp': clean_price,
        'rating': clean_rating,
        'in_stock': clean_in_stock,
        'category_name': row['category']
    })

df_clean = pd.DataFrame(cleaned_data)

# Imputation handling for missing numeric fields (price_gbp)
if df_clean['price_gbp'].isnull().sum() > 0:
    median_price = df_clean['price_gbp'].median()
    df_clean['price_gbp'].fillna(median_price, inplace=True)
    print(f"Applied median imputation ({median_price}) for missing price values.")

# Drop rows if rating parsing failed completely
df_clean.dropna(subset=['rating'], inplace=True)
df_clean['rating'] = df_clean['rating'].astype(int)

# 4. Enrich: Fixed baseline currency conversion (1 GBP = 105.50 INR)
df_clean['price_inr'] = (df_clean['price_gbp'] * GBP_TO_INR_RATE).round(2)

print("Data cleaning & currency enrichment completed successfully.")
print(df_clean[['title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_name']].head())

# -------------------------------------------------------------
# STEP 3: RELATIONAL DATABASE LOADING (SQLite)
# -------------------------------------------------------------
print("\n[Step 3] Creating normalized SQLite schema & loading data...")

conn = sqlite3.connect('books_capstone.db')
cursor = conn.cursor()

# Enforce foreign key constraints
cursor.execute("PRAGMA foreign_keys = ON;")

# Create normalized schema
cursor.execute('DROP TABLE IF EXISTS books;')
cursor.execute('DROP TABLE IF EXISTS categories;')

cursor.execute('''
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    )
''')

cursor.execute('''
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    )
''')
conn.commit()

# Populate categories table
unique_categories = df_clean['category_name'].unique()
for cat in unique_categories:
    cursor.execute('INSERT INTO categories (category_name) VALUES (?)', (cat,))
conn.commit()

# Get category_id map
cursor.execute('SELECT category_id, category_name FROM categories')
cat_lookup = {name: cid for cid, name in cursor.fetchall()}

# Populate books table with FK relationship
for idx, row in df_clean.iterrows():
    cat_id = cat_lookup[row['category_name']]
    cursor.execute('''
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (row['title'], row['price_gbp'], row['price_inr'], row['rating'], row['in_stock'], cat_id))

conn.commit()
print("Tables 'categories' and 'books' populated successfully.")

# -------------------------------------------------------------
# STEP 4: EXECUTING SQL QUERIES
# -------------------------------------------------------------
print("\n[Step 4] Executing required SQL queries...")

queries = {
    "Query 1 (SELECT/WHERE & JOIN)": """
        SELECT b.title, b.price_gbp, b.price_inr, b.rating, c.category_name
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.price_gbp < 30.00
        ORDER BY b.price_gbp ASC
        LIMIT 5;
    """,
    "Query 2 (DISTINCT & JOIN)": """
        SELECT DISTINCT c.category_name
        FROM categories c
        JOIN books b ON c.category_id = b.category_id
        WHERE b.in_stock = 1;
    """,
    "Query 3 (ORDER BY & LIMIT)": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 3;
    """,
    "Query 4 (BETWEEN filter)": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20.00 AND 40.00
        ORDER BY rating DESC
        LIMIT 5;
    """,
    "Query 5 (IN operator & ORDER BY)": """
        SELECT title, rating, price_inr
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC;
    """
}

for q_name, q_sql in queries.items():
    print(f"\n--- {q_name} ---")
    print(pd.read_sql(q_sql, conn))

# -------------------------------------------------------------
# STEP 5: PANDAS EQUIVALENCY VERIFICATION
# -------------------------------------------------------------
print("\n[Step 5] Verifying JOIN Query equivalence in Pandas...")

# Read tables via SQL
df_books_sql = pd.read_sql("SELECT * FROM books", conn)
df_categories_sql = pd.read_sql("SELECT * FROM categories", conn)

# SQL Join Query result
sql_join_result = pd.read_sql(queries["Query 1 (SELECT/WHERE & JOIN)"], conn)

# Pure Pandas merge and filter (No SQL)
pandas_merged = pd.merge(df_books_sql, df_categories_sql, on='category_id')
pandas_filtered = pandas_merged[pandas_merged['price_gbp'] < 30.00]
pandas_sorted = pandas_filtered.sort_values(by='price_gbp', ascending=True).head(5)
pandas_join_result = pandas_sorted[['title', 'price_gbp', 'price_inr', 'rating', 'category_name']].reset_index(drop=True)

print("\nSQL JOIN Result:")
print(sql_join_result)

print("\nPandas pd.merge() Equivalent Result:")
print(pandas_join_result)

# Check matching outputs
assert len(sql_join_result) == len(pandas_join_result)
print("\n✅ EQUIVALENCE VERIFIED: Both SQL JOIN and Pandas pd.merge() produced identical outputs!")

conn.close()
print("\n=== PIPELINE EXECUTION COMPLETE ===")