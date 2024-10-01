import os
import pandas as pd
import pdfplumber

def make_unique(columns):
    """Make the column names unique by appending a suffix if duplicates exist."""
    seen = {}
    unique_columns = []
    for column in columns:
        if column in seen:
            seen[column] += 1
            unique_columns.append(f"{column}_{seen[column]}")
        else:
            seen[column] = 0
            unique_columns.append(column)
    return unique_columns

def extract_cis_controls_tables(pdf_path, component_name):
    combined_data = []
    capture_tables = False

    with pdfplumber.open(pdf_path) as pdf:
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            text = page.extract_text()

            # Check if the page contains 'CIS Controls:' indicating table of interest
            if "CIS Controls:" in text:
                capture_tables = True

            if capture_tables:
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:  # Ensure the table has content
                        # Ensure the first row has unique column names
                        columns = make_unique(table[0])
                        
                        # Create DataFrame only if there are valid rows
                        valid_rows = [row for row in table[1:] if len(row) == len(columns)]
                        
                        if valid_rows:  # Only create DataFrame if valid rows exist
                            df = pd.DataFrame(valid_rows, columns=columns)
                            df['Component'] = component_name  # Add the component (PDF name)
                            combined_data.append(df)

                # Stop capturing if the next page is empty or irrelevant
                if i + 1 < len(pdf.pages) and not pdf.pages[i + 1].extract_text().strip():
                    break

    # Combine all the extracted DataFrames into a single DataFrame
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no tables found

def extract_from_folder(folder_path):
    combined_data = []
    
    # Iterate over all PDF files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file_name)
            component_name = os.path.splitext(file_name)[0]  # Use file name without extension as component name
            print(f"Extracting data from {file_name}")
            
            # Extract tables from the PDF
            df = extract_cis_controls_tables(pdf_path, component_name)
            
            if not df.empty:
                combined_data.append(df)

    # Combine all extracted data into a single DataFrame
    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no tables were found

def save_to_excel(dataframe, output_excel_path):
    """Save the DataFrame to an Excel file."""
    dataframe.to_excel(output_excel_path, index=False)

# Example usage
folder_path = 'path/to/your/folder'  # Replace with your folder path containing PDFs
output_excel_path = 'combined_cis_controls_tables_with_component.xlsx'  # Output Excel file

combined_df = extract_from_folder(folder_path)

# Check if combined_df is empty and print a message
if combined_df.empty:
    print("No tables found under 'CIS Controls:' in any PDF")
else:
    save_to_excel(combined_df, output_excel_path)
    print(f"Combined tables saved to {output_excel_path}")
