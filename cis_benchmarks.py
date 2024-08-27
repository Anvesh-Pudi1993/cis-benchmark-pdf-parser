import os
import PyPDF2
import re
import tkinter as tk
from tkinter import filedialog
import openpyxl
import pandas as pd

# Define the list of keywords to search for
keywords = ["description", "rationale", "impact", "audit", "remediation", "cis control"]

def extract_text_from_pdf(pdf_path, txt_path, excel_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            workbook = openpyxl.Workbook()
            worksheet = workbook.active

            # Construct new keywords without ": " and replace spaces with underscores
            new_keywords = [keyword.replace(": ", "").replace(" ", "_") for keyword in keywords]

            # Add column headers as the first row
            worksheet.append(["Num Page and rule"] + new_keywords)

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

                # Add the extracted text to the worksheet
                max_rows = max(len(keyword_data[keyword]) for keyword in keywords)
                for row in range(max_rows):
                    values = [first_two_lines_text]
                    for keyword in keywords:
                        keyword_column = keyword.replace(" ", "_")
                        if row < len(keyword_data[keyword]):
                            values.append(keyword_data[keyword][row].strip())
                        else:
                            values.append("")
                    worksheet.append(values)

            workbook.save(excel_path)
            print(f"Text extracted from '{pdf_path}' and saved to '{excel_path}'.")
    except Exception as e:
        print(f"Error: {e}")

def browse_pdf():
    for pdf_file in os.listdir(file_path):
        if pdf_file.endswith(".pdf"):
            file_path = os.path.join(file_path, pdf_file)
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        txt_file_path = os.path.splitext(file_path)[0] + ".txt"
        excel_file_path = os.path.splitext(file_path)[0] + ".xlsx"
        extract_text_from_pdf(file_path, txt_file_path, excel_file_path)

# Create a file explorer window
file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
if file_path:
    txt_file_path = os.path.splitext(file_path)[0] + ".txt"
    excel_file_path = os.path.splitext(file_path)[0] + ".xlsx"
    extract_text_from_pdf(file_path, txt_file_path, excel_file_path)


