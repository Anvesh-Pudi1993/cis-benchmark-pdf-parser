import fitz  # PyMuPDF
import pandas as pd
import os
import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    document = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        full_text += page.get_text()
    document.close()
    return full_text

def extract_cis_controls_table(text):
    """Extracts the CIS Controls table from the text."""
    tables = []
    cis_controls_section = re.search(r'CIS Controls[:\s\w]*', text, re.IGNORECASE)
    if not cis_controls_section:
        return tables

    # Extract text starting from the "CIS Controls" section
    start_index = cis_controls_section.end()
    text_after_section = text[start_index:]
    
    # This regex is a placeholder; adjust based on the exact format of your tables
    table_pattern = re.compile(r'(\d+)\s+(.*)\s+(.*)\s+(.*)\s*(.*)')  # Example regex pattern
    
    lines = text_after_section.split('\n')
    for line in lines:
        match = table_pattern.match(line.strip())
        if match:
            control_number, control_description, *additional_columns = match.groups()
            tables.append([control_number, control_description] + additional_columns)
    
    return tables

def main(pdf_directory, csv_file_path):
    """Main function to process all PDFs in a directory and save to CSV."""
    all_tables = []

    for pdf_filename in os.listdir(pdf_directory):
        if pdf_filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, pdf_filename)
            print(f"Processing {pdf_path}...")
            pdf_text = extract_text_from_pdf(pdf_path)
            tables = extract_cis_controls_table(pdf_text)
            all_tables.extend(tables)

    # Define columns based on the expected structure of the table
    columns = ['Control Number', 'Control Description', 'Additional Column 1', 'Additional Column 2', 'Additional Column 3']
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_tables, columns=columns)
    df.to_csv(csv_file_path, index=False)

    print(f"Data has been saved to {csv_file_path}")

if __name__ == "__main__":
    # Set your PDF directory and output CSV file path
    pdf_directory = 'C:/Users/Anvesh Pudi/Downloads/CIS_Benchmark/cis-benchmark-pdf-parser'
    csv_file_path = 'C:/Users/Anvesh Pudi/Downloads/CIS_Benchmark/cis-benchmark-pdf-parser/cis_controls_tables.csv'
    
    main(pdf_directory, csv_file_path)
