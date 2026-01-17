#!/usr/bin/env python3
"""
GroundZero AI - Tools Demo
==========================

Demonstrates:
1. Code Execution (Python & Bash)
2. Document Understanding (Read & Answer Questions)
3. File Creation (Word, Excel, PDF, PowerPoint)
4. Data Analysis

This is exactly what you need for your analytics work!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from groundzero import GroundZeroAI


def demo_code_execution(ai):
    """Demo: Execute Python and Bash code."""
    print("\n" + "="*60)
    print("📟 CODE EXECUTION DEMO")
    print("="*60)
    
    # Python: Simple calculation
    print("\n1. Python calculation:")
    result = ai.run_code("""
import math
radius = 5
area = math.pi * radius ** 2
print(f"Circle area with radius {radius}: {area:.2f}")
""")
    print(f"   Output: {result['output'].strip()}")
    
    # Python: Data processing
    print("\n2. Python data processing:")
    result = ai.run_code("""
data = [
    {"store": "Store A", "sales": 15000},
    {"store": "Store B", "sales": 22000},
    {"store": "Store C", "sales": 18500},
]
total = sum(d['sales'] for d in data)
avg = total / len(data)
print(f"Total sales: R{total:,.2f}")
print(f"Average: R{avg:,.2f}")
print(f"Best performer: {max(data, key=lambda x: x['sales'])['store']}")
""")
    print(f"   Output:\n   {result['output'].strip().replace(chr(10), chr(10) + '   ')}")
    
    # Bash command
    print("\n3. Bash command:")
    result = ai.run_code("echo 'Hello from bash!' && date", language="bash")
    print(f"   Output: {result['output'].strip()}")


def demo_document_understanding(ai):
    """Demo: Read and understand documents."""
    print("\n" + "="*60)
    print("📄 DOCUMENT UNDERSTANDING DEMO")
    print("="*60)
    
    # Create a sample CSV for demo
    sample_csv = ai.tools.workspace / "sales_data.csv"
    sample_csv.write_text("""Store,Region,Month,Sales,Units
Johannesburg Central,Gauteng,January,125000,850
Cape Town V&A,Western Cape,January,98000,720
Durban Gateway,KwaZulu-Natal,January,87500,640
Pretoria East,Gauteng,January,76000,520
Johannesburg Central,Gauteng,February,138000,920
Cape Town V&A,Western Cape,February,105000,780
Durban Gateway,KwaZulu-Natal,February,92000,680
Pretoria East,Gauteng,February,82000,560
Johannesburg Central,Gauteng,March,142000,950
Cape Town V&A,Western Cape,March,112000,820
Durban Gateway,KwaZulu-Natal,March,95000,700
Pretoria East,Gauteng,March,88000,600
""")
    
    print(f"\n1. Reading CSV file: {sample_csv.name}")
    doc_info = ai.read_document(str(sample_csv))
    
    if doc_info['success']:
        print(f"   ✓ Loaded: {doc_info['word_count']} words")
        print(f"   ✓ Tables found: {doc_info['tables']}")
        print(f"   ✓ Key topics: {', '.join(doc_info['key_topics'][:5])}")
    
    # Ask questions about the document
    print("\n2. Asking questions about the data:")
    
    questions = [
        "What was the total sales in Gauteng?",
        "Which store had the highest sales?",
        "What was the average monthly sales?",
    ]
    
    for q in questions:
        print(f"\n   Q: {q}")
        answer = ai.ask_documents(q)
        print(f"   A: {answer[:200]}...")
    
    # Clean up
    sample_csv.unlink()


def demo_file_creation(ai):
    """Demo: Create various file types."""
    print("\n" + "="*60)
    print("📝 FILE CREATION DEMO")
    print("="*60)
    
    workspace = ai.tools.files.output_dir
    
    # Create Excel file
    print("\n1. Creating Excel file...")
    try:
        data = [
            {"Product": "Widget A", "Q1": 15000, "Q2": 18000, "Q3": 22000, "Q4": 25000},
            {"Product": "Widget B", "Q1": 12000, "Q2": 14000, "Q3": 16000, "Q4": 19000},
            {"Product": "Widget C", "Q1": 8000, "Q2": 9500, "Q3": 11000, "Q4": 13000},
        ]
        filepath = ai.create_excel("quarterly_sales.xlsx", data)
        print(f"   ✓ Created: {filepath}")
    except Exception as e:
        print(f"   ⚠ Skipped (install openpyxl): {e}")
    
    # Create Word document
    print("\n2. Creating Word document...")
    try:
        content = [
            {"type": "heading", "text": "Sales Report Q4 2025", "level": 1},
            {"type": "paragraph", "text": "This report summarizes our quarterly performance."},
            {"type": "heading", "text": "Key Highlights", "level": 2},
            {"type": "list", "items": [
                "Revenue up 15% year-over-year",
                "New customer acquisition increased 20%",
                "Customer retention rate at 92%"
            ]},
            {"type": "heading", "text": "Regional Performance", "level": 2},
            {"type": "table", "headers": ["Region", "Sales", "Growth"],
             "rows": [
                 ["Gauteng", "R2.5M", "+18%"],
                 ["Western Cape", "R1.8M", "+12%"],
                 ["KZN", "R1.2M", "+8%"],
             ]},
        ]
        filepath = ai.create_word("sales_report.docx", content, title="Q4 Sales Report")
        print(f"   ✓ Created: {filepath}")
    except Exception as e:
        print(f"   ⚠ Skipped (install python-docx): {e}")
    
    # Create CSV
    print("\n3. Creating CSV file...")
    try:
        data = [
            ["Date", "Store", "Sales", "Transactions"],
            ["2025-01-15", "JHB Central", "45000", "120"],
            ["2025-01-15", "CPT V&A", "38000", "95"],
            ["2025-01-15", "DBN Gateway", "32000", "88"],
        ]
        result = ai.tools.create_csv("daily_sales.csv", data)
        if result.success:
            print(f"   ✓ Created: {result.result}")
    except Exception as e:
        print(f"   ⚠ Error: {e}")
    
    # Create PowerPoint
    print("\n4. Creating PowerPoint...")
    try:
        slides = [
            {"type": "title", "title": "Monthly Review", "subtitle": "January 2026"},
            {"type": "content", "title": "Achievements", "content": [
                "Exceeded sales target by 12%",
                "Launched 3 new product lines",
                "Expanded to 2 new regions"
            ]},
            {"type": "content", "title": "Next Steps", "content": [
                "Focus on customer retention",
                "Optimize supply chain",
                "Invest in staff training"
            ]},
        ]
        filepath = ai.create_powerpoint("monthly_review.pptx", slides)
        print(f"   ✓ Created: {filepath}")
    except Exception as e:
        print(f"   ⚠ Skipped (install python-pptx): {e}")
    
    # List created files
    print(f"\n5. Files in workspace:")
    for f in workspace.glob("*"):
        print(f"   - {f.name} ({f.stat().st_size:,} bytes)")


def demo_data_analysis(ai):
    """Demo: Data analysis capabilities."""
    print("\n" + "="*60)
    print("📊 DATA ANALYSIS DEMO")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample retail data...")
    result = ai.run_code("""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample retail data
np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=90, freq='D')
stores = ['JHB Central', 'CPT V&A', 'DBN Gateway', 'PTA East']

data = []
for date in dates:
    for store in stores:
        base_sales = {'JHB Central': 50000, 'CPT V&A': 40000, 
                      'DBN Gateway': 35000, 'PTA East': 30000}[store]
        # Add some randomness and weekly pattern
        day_factor = 1.2 if date.dayofweek in [4, 5] else 1.0  # Weekend boost
        sales = base_sales * day_factor * np.random.uniform(0.8, 1.3)
        data.append({
            'Date': date,
            'Store': store,
            'Sales': round(sales, 2),
            'Transactions': int(sales / 400)
        })

df = pd.DataFrame(data)
df.to_csv('retail_data.csv', index=False)
print(f"Created retail_data.csv with {len(df)} rows")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Stores: {df['Store'].unique().tolist()}")
""")
    print(f"   {result['output'].strip()}")
    
    # Analyze the data
    print("\n2. Running analysis...")
    result = ai.run_code("""
import pandas as pd

df = pd.read_csv('retail_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("=== SALES SUMMARY ===")
print(f"Total Sales: R{df['Sales'].sum():,.2f}")
print(f"Total Transactions: {df['Transactions'].sum():,}")
print()

print("=== BY STORE ===")
store_summary = df.groupby('Store').agg({
    'Sales': ['sum', 'mean'],
    'Transactions': 'sum'
}).round(2)
store_summary.columns = ['Total Sales', 'Avg Daily', 'Total Trans']
store_summary = store_summary.sort_values('Total Sales', ascending=False)
print(store_summary.to_string())
print()

print("=== MONTHLY TREND ===")
df['Month'] = df['Date'].dt.to_period('M')
monthly = df.groupby('Month')['Sales'].sum()
for month, sales in monthly.items():
    print(f"  {month}: R{sales:,.2f}")
""")
    print(f"   Output:\n   {result['output'].strip().replace(chr(10), chr(10) + '   ')}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              🧠 GROUNDZERO AI - TOOLS DEMO                    ║
║                                                               ║
║  State-of-the-art capabilities for your analytics work:      ║
║  • Code Execution (Python & Bash)                            ║
║  • Document Understanding (Read ANY file, Answer Questions)  ║
║  • File Creation (Word, Excel, PDF, PowerPoint)             ║
║  • Data Analysis & Visualization                             ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Initialize AI
    print("Initializing GroundZero AI...")
    ai = GroundZeroAI(use_mock=True)
    print("✓ Ready!\n")
    
    # Run demos
    demo_code_execution(ai)
    demo_file_creation(ai)
    demo_data_analysis(ai)
    # demo_document_understanding(ai)  # Uncomment after loading a real document
    
    print("\n" + "="*60)
    print("✓ DEMO COMPLETE!")
    print("="*60)
    print("""
To use in your analytics work:

1. Read documents:
   ai.read_document("sales_report.pdf")
   ai.read_document("data.xlsx")
   ai.read_documents(["file1.csv", "file2.xlsx", "report.pdf"])

2. Ask questions about content:
   ai.ask_documents("What was the total revenue?")
   ai.ask_documents("Which region performed best?")
   ai.ask_documents("Summarize the key findings")

3. Run analysis code:
   ai.run_code("import pandas as pd; df = pd.read_csv('data.csv'); print(df.describe())")

4. Create reports:
   ai.create_excel("output.xlsx", data)
   ai.create_word("report.docx", content)
   ai.create_pdf("summary.pdf", content)
""")


if __name__ == "__main__":
    main()
