import re
import pandas as pd
df1=pd.read_excel('CSI_clean_Verify.xlsx',sheet_name="Guidelines+Description")
def strip_numbers(sentence):
    # Define the regex pattern for the numbers at the beginning
    pattern = r'^\.\d+(?:\.\d+)* '

    # Remove the numbers pattern from the beginning of the sentence
    cleaned_sentence = re.sub(pattern, '', sentence)
    
    return cleaned_sentence
df1['Num Page and rule']=df1['Num Page and rule'].apply(strip_numbers)
df1.to_excel('CSI_clean_Verify_V1.xlsx')
