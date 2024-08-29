import pdfplumber
import pandas as pd
import os

# Define the folder path containing the PDFs
folder_path = 'C:/Users/Anvesh Pudi/Downloads/CIS_Benchmark/Test'  # Update with your actual folder path

# Function to extract tables with "CIS Controls:" heading
def extract_cis_controls_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if  "CIS Controls:" in text:
                # Extract tables on the page
                page_tables = page.extract_tables()
                for table in page_tables:
                    # Find column headers and extract "Controls Version" and "Control"
                    headers = table[0]  # Assumes the first row is the header
                    if "Controls Version" in headers and "Control" in headers:
                        # Find indices of required columns
                        version_idx = headers.index("Controls Version")
                        control_idx = headers.index("Control")
                        
                        # Extract required data from the table
                        for row in table[1:]:  # Skip the header row
                            version = row[version_idx] if version_idx < len(row) else None
                            control = row[control_idx] if control_idx < len(row) else None
                            tables.append([version, control])
    return tables

# Main function to iterate over PDFs and save data to Excel
def extract_data_to_excel(folder_path, output_excel='output.xlsx'):
    all_data = []  # To store data from all PDFs

    # Iterate over all PDF files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            data = extract_cis_controls_tables(pdf_path)
            all_data.extend(data)

    # Convert extracted data to a pandas DataFrame
    df = pd.DataFrame(all_data, columns=['Controls Version', 'Control'])

    # Save the DataFrame to an Excel file
    df.to_excel(output_excel, index=False)
    print(f"Data saved to {output_excel}")

# Run the extraction process
extract_data_to_excel(folder_path)
