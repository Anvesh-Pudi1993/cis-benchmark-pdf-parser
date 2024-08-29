import re
import pandas as pd
df=pd.read_excel('CIS_Benchmarks_Cleaned_updated.xlsx')
def clean_sentence(sentence):
     sentence = re.sub(r'^(Page\s*\d+(\.\d+)?\s*|\d+(\.\d+)*\s*|(\.\d+)+\s*)', '', sentence, flags=re.IGNORECASE)
    # Remove everything before and including the word 'Ensure', if present
     ensure_match = re.search(r'\bEnsure\b', sentence, flags=re.IGNORECASE)
     if ensure_match:
        # Keep the part of the sentence starting from 'Ensure'
        return sentence[ensure_match.start():].strip()
     else:
        # If 'Ensure' is not found, strip leading numbers
        return re.sub(r'^\d+\s*&^\.\d+\.\d+\.\d+&^\.\d+', '', sentence)
df['Num Page and rule']=df['Num Page and rule'].apply(clean_sentence)
df.to_excel('CSI_clean.xlsx')

