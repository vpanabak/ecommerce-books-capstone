# Module 1: Catalog Data Pipeline & Benchmarking

This directory contains the live web scraping, data cleaning, currency enrichment, and relational storage pipeline for competitive catalog benchmarking.

## Data Source & Scope
* **Target:** `http://books.toscrape.com/` (Public scraping practice platform).
* **Scope:** Scrapes a minimum of 60 books across 3 categories.
* **Fields Captured:** Title, Price (GBP), Star Rating (Text), Availability, Category.

---

## Data Cleaning & Parsing Decisions
1. **Price (`price_gbp`):** Stripped the `£` and non-ASCII currency encoding characters. Converted values to `float`.
2. **Missing/Malformed Values:** Any unparseable numeric price values default to median imputation across the dataset so execution continues smoothly without crashing. Unparseable ratings are dropped.
3. **Star Rating (`rating`):** Mapped text rating strings (`One`, `Two`, `Three`, `Four`, `Five`) to integer values (`1` through `5`).
4. **Availability (`in_stock`):** Parsed text availability string into a binary integer/boolean (`1` for in stock, `0` for out of stock).
5. **Fixed Baseline Currency Conversion:** Converted `price_gbp` to `price_inr` using the strict project baseline constant:
   $$\text{1 GBP} = \text{105.50 INR}$$
   *This is a fixed, project-defined constant with no live market lookup or date reference.*

---

## Normalized Database Schema
The database uses SQLite (`books_capstone.db`) with a 2-table primary key / foreign key architecture:

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);