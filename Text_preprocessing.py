import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data files (ensure all are downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Sometimes needed for lemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load your data from Excel
input_file = 'CIS_Benchmarks_Consolidated_Cleaned.xlsx'  # Replace with your input file name
output_file = 'output_cleaned.xlsx'

# Load data into pandas DataFrame
df = pd.read_excel(input_file)

# Define text preprocessing function
def clean_text(text):
    if isinstance(text, str):  # Check if the value is a string
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize text
        words = word_tokenize(text)
        # Remove stopwords and perform lemmatization
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        # Join words back into a single string
        return ' '.join(words)
    else:
        return text  # If not a string, return as-is

# Apply text cleaning function to all columns
for column in df.columns:
    df[column] = df[column].apply(clean_text)

# Save the cleaned data to a new Excel file
df.to_excel(output_file, index=False)

print(f"Cleaned data has been saved to {output_file}.")
