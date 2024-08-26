import fitz  # PyMuPDF
import re
import csv

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def extract_ics_controls(text):
    """
    Extracts ICS controls information from the text.
    """
    ics_controls = []
    # Assuming ICS controls are denoted in a specific pattern. Adjust regex as needed.
    pattern = re.compile(r'\bICS Control\b.*?(?=\n\n|\Z)', re.DOTALL)
    matches = pattern.findall(text)
    
    for match in matches:
        # Clean and process the match if needed
        ics_controls.append(match.strip())
    
    return ics_controls

def save_to_csv(data, output_file):
    """
    Saves extracted ICS controls information to a CSV file.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ICS Control'])
        for row in data:
            writer.writerow([row])

def main():
    pdf_path = 'CIS Docker Benchmark v1.7 PDF.pdf'  # Update this path
    output_csv_path = 'ics_controls.csv'  # Output CSV file path
    
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Extract ICS controls from the text
    ics_controls = extract_ics_controls(text)
    
    # Save the ICS controls to a CSV file
    save_to_csv(ics_controls, output_csv_path)
    
    print(f"ICS controls have been successfully saved to {output_csv_path}")

if __name__ == "__main__":
    main()
