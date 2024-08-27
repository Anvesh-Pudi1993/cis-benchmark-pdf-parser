import os
import PyPDF2
import pandas as pd
import re
from tkinter import filedialog

def extract_cis_controls_info(pdf_path):
    extracted_data = []
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            component_name = os.path.splitext(os.path.basename(pdf_path))[0]

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Extract the first line as the heading
                first_line = text.split('\n', 1)[0].strip()

                # Search for the first row of the CIS controls table
                # This assumes that the first row of the table contains control version and control info
                # Adjust the regular expression as per the structure of the table
                table_data = re.findall(r'(\d+\.\d+)\s+([A-Za-z0-9\s\-]+)\s+(v[\d\.]+)', text)

                if table_data:
                    # Only take the first row of the table
                    control_version, control_info, control_id = table_data[0]
                    extracted_data.append([component_name, first_line, f"{control_version} - {control_info}", control_id])
                    break  # Exit after finding the first relevant row

    except Exception as e:
        print(f"Error processing '{pdf_path}': {e}")

    return extracted_data

def process_pdfs_in_folder(folder_path):
    all_data = []
    headers = ["Component Name", "Heading", "ICS Control Info", "Control Version"]

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith((".pdf", ".PDF")):
            pdf_path = os.path.join(folder_path, pdf_file)
            pdf_data = extract_cis_controls_info(pdf_path)
            all_data.extend(pdf_data)

    # Convert the collected data to a DataFrame
    df = pd.DataFrame(all_data, columns=headers)

    # Save the DataFrame to an Excel file
    output_excel_path = os.path.join(folder_path, "CIS_Benchmarks_ICS_Controls_Info.xlsx")
    df.to_excel(output_excel_path, index=False)
    print(f"ICS control info extracted and saved to '{output_excel_path}'.")

def browse_folder_and_process_pdfs():
    folder_path = filedialog.askdirectory(title="Select Folder Containing PDF Files")
    if folder_path:
        process_pdfs_in_folder(folder_path)

# Call the function to browse and process PDFs
browse_folder_and_process_pdfs()
