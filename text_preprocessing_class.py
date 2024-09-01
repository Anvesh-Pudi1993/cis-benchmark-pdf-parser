import re
import pandas as pd
df=pd.read_excel('CIS_Benchmarks_Cleaned_updated.xlsx')
class text_preprocessing:
    def __init__(self,df):
        self.df=df

    def clean_sentence(self,sentence):
        sentence = re.sub(r'^(Page\s*\d+(\.\d+)?\s*|\d+(\.\d+)*\s*|(\.\d+)+\s*)', '', sentence, flags=re.IGNORECASE)
    # Remove everything before and including the word 'Ensure', if present
        ensure_match = re.search(r'\bEnsure\b', sentence, flags=re.IGNORECASE)
        if ensure_match:
        # Keep the part of the sentence starting from 'Ensure'
          return sentence[ensure_match.start():].strip()
        else:
        # If 'Ensure' is not found, strip leading numbers
          return re.sub(r'^\d+\s*&^\.\d+\.\d+\.\d+&^\.\d+', '', sentence)
     
    def strip_numbers(self):
    # Define the regex pattern for the numbers at the beginning
        pattern = r'^\.\d+(?:\.\d+)*'
    # Remove the numbers pattern from the beginning of the sentence
        for col in self.df.columns:
           self.df[col] = self.df[col].astype(str).apply(self.clean_sentence)
           self.df[col] = self.df[col].apply(lambda x: re.sub(pattern, '', x))
        #    self.df[col]=re.sub(pattern, '', self.df[col])

        self.df.to_excel('cis_class.xlsx')
processor = text_preprocessing(df)
processor.strip_numbers()

 



    
        
     
    
  