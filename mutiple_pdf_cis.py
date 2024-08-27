import os
import PyPDF2
import pandas as pd
from collections import defaultdict

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
    return text

# Function to parse the required information from the extracted text
def parse_cis_benchmark(text):
    sections = ["Description", "Rationale", "Impact", "Audit", "Remediation", "CIS Controls"]
    data = defaultdict(list)
    current_section = None
    
    for page in text:
        lines = page.split('\n')
        for line in lines:
            # Check if the line contains a section header
            if any(section in line for section in sections):
                section_name = line.split(":")[0].strip()  # Get the section name
                section_content = line.split(":")[1].strip() if ":" in line else ""
                current_section = section_name
                data[current_section].append(section_content)
            # Add to the current section if a section header was found
            elif current_section:
                data[current_section][-1] += " " + line.strip()
    
    return data

# Ensure all lists in the dictionary have the same length
def equalize_list_lengths(data_dict):
    max_len = max(len(v) for v in data_dict.values())
    for key in data_dict:
        while len(data_dict[key]) < max_len:
            data_dict[key].append(None)  # or '' to pad with empty strings
    return data_dict

# Function to process all PDF files in a folder
def process_pdf_folder(folder_path):
    all_data = []
    
    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            pdf_title = text[0].split('\n')[0]  # Assuming the title is on the first line of the first page
            parsed_data = parse_cis_benchmark(text)
            parsed_data = equalize_list_lengths(parsed_data)  # Ensure all lists are the same length
            parsed_data['Title'] = [pdf_title] * len(parsed_data[next(iter(parsed_data))])
            all_data.append(pd.DataFrame(parsed_data))
    
    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# Path to the folder containing the PDF files
pdf_folder_path = 'C:/Users/Anvesh Pudi/Downloads/CIS_Benchmark/CIS_Pdfs'

# Process the PDFs and get the DataFrame
final_df = process_pdf_folder(pdf_folder_path)

# Save the DataFrame to an Excel file
final_df.to_excel("CIS_Benchmarks_Consolidated.xlsx", index=False)

print("Data has been extracted and saved to 'CIS_Benchmarks_Consolidated.xlsx'.")