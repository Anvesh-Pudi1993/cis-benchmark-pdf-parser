import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load the Excel files
file1_path = 'path_to_your_first_excel.xlsx'  # Replace with your file path
file2_path = 'path_to_your_second_excel.xlsx'  # Replace with your file path

# Read the Excel files
df1 = pd.read_excel(file1_path)  # File with "component name", "Guidelines", and "description"
df2 = pd.read_excel(file2_path)  # File with "Control domain" and "control statement"

# Extract relevant columns
component_name = df1['component name'].apply(lambda x: x.strip())
guidelines = df1['Guidelines'].apply(lambda x: x.strip())
description = df1['description'].apply(lambda x: x.strip())

control_domain = df2['Control domain'].apply(lambda x: x.strip())
control_statement = df2['control statement'].apply(lambda x: x.strip())

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

# Function to process texts in smaller chunks to avoid memory issues
def process_in_chunks(guidelines, control_domains, chunk_size=50):
    """Process texts in chunks to avoid memory issues."""
    matched_results = []
    
    for i in tqdm(range(0, len(control_domains), chunk_size)):
        chunk_control_domains = control_domains[i:i+chunk_size]
        chunk_control_statements = control_statement[i:i+chunk_size]

        # Tokenize and encode inputs
        inputs1 = tokenizer(guidelines, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs2 = tokenizer(chunk_control_domains, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
            
            # Extract embeddings for the [CLS] token
            embeddings1 = outputs1.last_hidden_state[:, 0, :].numpy()
            embeddings2 = outputs2.last_hidden_state[:, 0, :].numpy()
            
            # Compute cosine similarity
            similarity_scores = cosine_similarity(embeddings1, embeddings2)
        
        # Find the most similar control domain for each guideline
        for idx1, scores in enumerate(similarity_scores):
            max_score_idx = scores.argmax()
            if scores[max_score_idx] > 0.5:  # Similarity threshold
                matched_results.append((
                    component_name.iloc[idx1], guidelines[idx1], description[idx1],
                    chunk_control_domains[max_score_idx], chunk_control_statements[max_score_idx], scores[max_score_idx]
                ))
    
    return matched_results

# Run the chunk processing and get the results
results = process_in_chunks(guidelines.tolist(), control_domain.tolist(), chunk_size=50)

# Create a DataFrame with the matched results, only including the specified columns
output_df = pd.DataFrame(results, columns=[
    'Component Name', 'Guidelines', 'Description', 
    'Control Domain', 'Control Statement'
])

# Save the output to an Excel file
output_df.to_excel('matched_results.xlsx', index=False)

print("Matching completed and saved to matched_results.xlsx")
