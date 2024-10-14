import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Load the Excel files
df1 = pd.read_excel('file1.xlsx')  # Contains "component name", "guidelines", "description"
df2 = pd.read_excel('file2.xlsx')  # Contains "Control domain", "standard statement"

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the embeddings of the [CLS] token
    return outputs.last_hidden_state[:, 0, :].numpy()

# Compute embeddings for descriptions and standard statements
df1['description_embedding'] = df1['description'].apply(get_embedding)
df2['standard_statement_embedding'] = df2['standard statement'].apply(get_embedding)

# Calculate similarity scores
results = []

for index, row in df2.iterrows():
    standard_statement_embedding = row['standard_statement_embedding']
    for _, desc_row in df1.iterrows():
        description_embedding = desc_row['description_embedding']
        # Calculate cosine similarity
        similarity = cosine_similarity(standard_statement_embedding, description_embedding)[0][0]
        results.append({
            'Control domain': row['Control domain'],
            'Standard statement': row['standard statement'],
            'Guidelines': desc_row['guidelines'],
            'Similarity Score': similarity
        })

# Create a DataFrame for results
results_df = pd.DataFrame(results)

# Display the results
print(results_df)

# Optionally, save the results to an Excel file
results_df.to_excel('similarity_results.xlsx', index=False)