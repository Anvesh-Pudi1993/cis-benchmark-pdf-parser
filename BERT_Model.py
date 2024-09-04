import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # For progress bars

# Load the Excel files
file1_path = 'MCL.xlsx'  # Replace with your file path
file2_path = 'CSI_clean_Verify_V1.xlsx'  # Replace with your file path

df1 = pd.read_excel(file1_path)  # The file with 820 rows
df2 = pd.read_excel(file2_path)  # The file with 29,000 rows

# Extract relevant columns
text_column1 = df1['Control domain'].apply(lambda x: x.strip())
text_column2 = df2['Guidelines'].apply(lambda x: x.strip())

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()

inputs1=tokenizer(text_column1.tolist(),renturn_tensors='pt',padding=True)
inputs2=tokenizer(text_column2.tolist(),renturn_tensors='pt',padding=True)

# Function to get BERT embeddings for a batch of text
def get_bert_embeddings_batch(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Use CLS token embeddings

# Generate BERT embeddings for df2 (batched)
batch_size = 32  # Adjust batch size based on your system's memory
embeddings2 = []
for i in tqdm(range(0, len(text_column2), batch_size)):
    batch_texts = text_column2[i:i + batch_size].tolist()
    batch_embeddings = get_bert_embeddings_batch(batch_texts)
    embeddings2.extend(batch_embeddings)

# Initialize lists to store matches and similarity scores
matches = []
similarity_scores = []

# Compare each row in df1 with the rows in df2
for i, text1 in enumerate(tqdm(text_column1)):
    emb1 = get_bert_embeddings_batch([text1])[0]  # Get embedding for text1
    best_similarity = 0
    best_match = None
    for j, emb2 in enumerate(embeddings2):
        similarity = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = df2.iloc[j]  # Store the matching row from df2
    if best_similarity > 0.5:  # Use a threshold to determine if the rows are similar
        matches.append(best_match)  # Append the best match to the list
        similarity_scores.append(best_similarity)  # Append the best similarity score

# Create a DataFrame for the results
results_df = pd.DataFrame(matches)
results_df['Similarity Score'] = similarity_scores

# Save the matched rows and similarity scores to a new Excel file
output_path = 'matched_results.xlsx'
results_df.to_excel(output_path, index=False)

print(f"Results saved to {output_path}")
