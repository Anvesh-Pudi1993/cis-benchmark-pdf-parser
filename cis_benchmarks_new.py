import os
import PyPDF2
import re
from tkinter import filedialog
import openpyxl
import pandas as pd

# Define the list of keywords to search for
keywords = ["description", "rationale", "impact", "audit", "remediation", "cis control"]

def extract_text_from_pdf(pdf_path):
    data = []
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            component_name = os.path.splitext(os.path.basename(pdf_path))[0]

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Extract the first two lines
                first_two_lines = text.split('\n', 2)[:2]
                first_two_lines_text = ' '.join(line.strip() for line in first_two_lines)

                # Search for text between keywords with context-aware parsing
                keyword_data = {}
                for i in range(len(keywords) - 1):
                    start_keyword = keywords[i]
                    end_keyword = keywords[i + 1]
                    pattern = re.compile(f'{start_keyword}(.*?){end_keyword}', re.DOTALL | re.IGNORECASE)
                    matches = pattern.findall(text)
                    keyword_data[start_keyword] = matches

                # Handle 'cis control' keyword separately
                cis_control_matches = re.findall(r'cis\s*control\s*\((.*?)\)', text, re.DOTALL | re.IGNORECASE)
                keyword_data['cis control'] = cis_control_matches

                # Collect the extracted text into the data list
                max_rows = max(len(keyword_data[keyword]) for keyword in keywords)
                for row in range(max_rows):
                    values = [component_name, first_two_lines_text]
                    for keyword in keywords:
                        if row < len(keyword_data[keyword]):
                            values.append(keyword_data[keyword][row].strip())
                        else:
                            values.append("")
                    data.append(values)

    except Exception as e:
        print(f"Error processing '{pdf_path}': {e}")

    return data

def process_pdfs_in_folder(folder_path):
    all_data = []
    
    # Column headers
    headers = ["Component Name", "Num Page and rule"] + [keyword.replace(": ", "").replace(" ", "_") for keyword in keywords]

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith((".pdf", ".PDF")):
            pdf_path = os.path.join(folder_path, pdf_file)
            pdf_data = extract_text_from_pdf(pdf_path)
            all_data.extend(pdf_data)

    # Convert the collected data to a DataFrame
    df = pd.DataFrame(all_data, columns=headers)

    # Save the DataFrame to an Excel file
    output_excel_path = os.path.join(folder_path, "CIS_Benchmarks_Consolidated.xlsx")
    df.to_excel(output_excel_path, index=False)
    print(f"All data extracted and saved to '{output_excel_path}'.")

def browse_folder_and_process_pdfs():
    folder_path = filedialog.askdirectory(title="Select Folder Containing PDF Files")
    if folder_path:
        process_pdfs_in_folder(folder_path)

# Call the function to browse and process PDFs
browse_folder_and_process_pdfs()