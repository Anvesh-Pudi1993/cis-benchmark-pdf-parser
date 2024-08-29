import re 
import pandas as pd
df=pd.read_excel('CIS_Benchmarks_Cleaned_updated.xlsx')
def preprocess_text(text):
    # Regex to remove the "Page XX ... General/Specific ... X.X.X" part

    # text = re.sub(r'Page \d+\s+Internal Only(?:\s+-\s+\w+\s+\d+\.\d+\.\d+)?', '', text)
    # text = re.sub(r'^.*?', '', text, flags=re.IGNORECASE)
    # re.sub(r'^\d+(\.\d+)*\s*', '', text)
    # re.sub(r'Page\s*\d+\s*', '', text)
    # text = re.sub(r'^Page \d+ Internal Only - General \d+\.\d+\.\d+ ', '', text)
    
    # Remove trailing colons
    text = re.sub(r':$', '', text)
    
    # Remove special characters (keeping only alphanumeric characters and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Regex to remove "• Level X" part
    # text = re.sub(r'•\s*Level \d+', '', text)
    
    # Strip leading and trailing whitespace

    ensure_match = re.search(r'\bEnsure\b', text, flags=re.IGNORECASE)
    if ensure_match:
        # Keep the part of the sentence starting from 'Ensure'
        return text[ensure_match.start():].strip()
    else:
        # If 'Ensure' is not found, strip leading numbers
        return re.sub(r'^\D*\d+\s*', '', text)
    return text.strip()
    
cols=['Num Page and rule','description','remediation','rationale','audit']
for col in cols:
    df['Num Page and rule']=df['Num Page and rule'].apply(preprocess_text)

df.to_excel('CSI_Benchmark_clean.xlsx')






