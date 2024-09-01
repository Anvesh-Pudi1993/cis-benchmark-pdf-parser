import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load the Excel files
file1_path = 'MCL.xlsx'  # Replace with your file path
file2_path = 'CSI_clean_Verify_V1.xlsx'  # Replace with your file path

df1 = pd.read_excel(file1_path)  # The file with 820 rows
df2 = pd.read_excel(file2_path)  # The file with 29,000 rows

# Extract relevant columns
text_column1 = df1['Control domain'].apply(lambda x: x.strip())
text_column2 = df2['Guidelines'].apply(lambda x: x.strip())

# Load DistilBERT model and tokenizer (smaller model for memory efficiency)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()

# Process data in chunks
def process_in_chunks(texts1, texts2, chunk_size=100):
    """Process texts in chunks to avoid memory issues."""
    similar_sentences = []
    
    for i in tqdm(range(0, len(texts2), chunk_size)):
        chunk_texts2 = texts2[i:i+chunk_size]
        
        # Tokenize and encode texts
        inputs1 = tokenizer(texts1, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs2 = tokenizer(chunk_texts2, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
            
            embeddings1 = outputs1.last_hidden_state[:, 0, :].numpy()
            embeddings2 = outputs2.last_hidden_state[:, 0, :].numpy()
            
            # Calculate cosine similarity
            similarity_scores = cosine_similarity(embeddings1, embeddings2)
        
        # Collect similar sentences above a threshold
        for idx1, scores in enumerate(similarity_scores):
            for idx2, score in enumerate(scores):
                if score > 0.5:  # Similarity threshold
                    similar_sentences.append((texts1[idx1], chunk_texts2[idx2], score))
    
    return similar_sentences

# Run the chunk processing
similar_sentences = process_in_chunks(text_column1.tolist(), text_column2.tolist(), chunk_size=50)

# Write results to Excel
output_df = pd.DataFrame(similar_sentences, columns=['Text_Column1', 'Text_Column2', 'Similarity_Score'])
output_df.to_excel('output.xlsx', index=False)

print("Comparison completed and saved to output.xlsx")

