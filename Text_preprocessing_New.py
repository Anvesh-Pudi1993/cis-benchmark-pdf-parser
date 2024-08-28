# Define a simpler function to clean the text without using NLTK
import re
import pandas as pd
df=pd.read_excel('CIS_Benchmarks_Consolidated_Cleaned.xlsx')
def simple_clean_text(text):
    # Remove patterns that start with 'page '+numeric+' internal only general'+numeric
    text = re.sub(r'page \d+ internal only general \d+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower().strip()
    
    return text

# Apply the cleaning function to relevant columns
for col in ['Component Name', 'Num Page and rule', 'description', 'rationale', 'impact', 'audit', 'remediation']:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(simple_clean_text)

# Display the cleaned dataframe
df.to_excel('cleaned_CISbookmarks.xlsx')
