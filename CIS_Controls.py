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

def extract_cis_controls_tables(pdf_path):
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

def save_to_excel(dataframe, output_excel_path):
    """Save the DataFrame to an Excel file."""
    dataframe.to_excel(output_excel_path, index=False)

# Example usage
pdf_path = 'C:/Users/Anvesh Pudi/Downloads/CIS_Benchmark/cis-benchmark-pdf-parser/CIS Docker Benchmark v1.7 PDF.pdf'  # Replace with your PDF file path
output_excel_path = 'combined_cis_controls_tables.xlsx'  # Output Excel file
combined_df = extract_cis_controls_tables(pdf_path)

# Check if combined_df is empty and print a message
if combined_df.empty:
    print("No tables found under 'CIS Controls:'")
else:
    save_to_excel(combined_df, output_excel_path)
    print(f"Combined tables saved to {output_excel_path}")
