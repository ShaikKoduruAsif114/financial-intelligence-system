import os
import random
from pathlib import Path
from fpdf import FPDF

DATA_DIR = Path('data/raw')
DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_pdf(text: str, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Encode text to avoid character encoding issues in FPDF
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    pdf.output(str(DATA_DIR / filename))

def main():
    print("Generating 150 PDF documents with synthetic financial data...")
    num_docs = 150
    
    companies = ["Acme Corp", "TechNova", "GlobalFinance", "NexGen Industries", "Alpha Holdings"]
    trends = ["increased", "decreased", "remained stable", "saw significant growth", "experienced a downturn"]
    
    for i in range(num_docs):
        company = random.choice(companies)
        trend = random.choice(trends)
        revenue = random.randint(10, 500)
        
        doc_text = f"FINANCIAL DOCUMENT {i+1}\n\n"
        doc_text += "CONFIDENTIAL - FOR INTERNAL USE ONLY\n\n"
        doc_text += f"Summary of Findings for {company}:\n"
        doc_text += f"In the latest quarter, the overall revenue {trend}, reaching ${revenue} million. "
        doc_text += "Operating margins were affected by supply chain adjustments and new investments in R&D.\n\n"
        doc_text += "Analysis:\nThis document contains forward-looking statements. "
        doc_text += "Actual results may vary. Past performance is not indicative of future results.\n"
        doc_text += "Major risk factors include market volatility and regulatory changes."
        
        filename = f"financial_doc_{i+1:03d}.pdf"
        create_pdf(doc_text, filename)
        
        if (i+1) % 25 == 0:
            print(f"Generated {i+1}/{num_docs} PDFs...")

    print(f"\nDone! Generated {num_docs} PDFs in {DATA_DIR.absolute()}")

if __name__ == "__main__":
    main()
