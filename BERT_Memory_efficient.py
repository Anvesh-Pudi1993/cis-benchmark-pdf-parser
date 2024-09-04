import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import gc

# Load the Excel files
file1_path = 'MCL.xlsx'  # Replace with your file path
file2_path = 'CSI_clean_Verify_V1.xlsx'  # Replace with your file path

# Read the Excel files
df1 = pd.read_excel(file2_path)  # File with "component name", "Guidelines", and "description"
df2 = pd.read_excel(file1_path)  # File with "Control domain" and "control statement"
print(df2.columns)
# Convert all values to strings and handle NaN values
component_name = df1['Component Name'].fillna('').astype(str).apply(lambda x: x.strip())
guidelines = df1['Guidelines'].fillna('').astype(str).apply(lambda x: x.strip())
description = df1['description'].fillna('').astype(str).apply(lambda x: x.strip())

control_domain = df2['Control domain'].fillna('').astype(str).apply(lambda x: x.strip())
control_statement = df2['Standard statement'].fillna('').astype(str).apply(lambda x: x.strip())

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

# Function to process texts in batches
def process_in_batches(guidelines, control_domains, batch_size=10):
    """Process texts in batches to avoid memory issues."""
    matched_results = []
    
    # Iterate through the guidelines in batches
    for batch_start in tqdm(range(0, len(guidelines), batch_size)):
        guideline_batch = guidelines[batch_start:batch_start + batch_size]
        component_name_batch = component_name[batch_start:batch_start + batch_size]
        description_batch = description[batch_start:batch_start + batch_size]
        
        # Tokenize and encode inputs for the current batch of guidelines
        inputs1 = tokenizer(guideline_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        # Iterate through control domains in batches to compare with the guideline batch
        for i in range(0, len(control_domains), batch_size):
            control_domain_batch = control_domains[i:i + batch_size]
            control_statement_batch = control_statement[i:i + batch_size]
            
            # Tokenize and encode inputs for the current batch of control domains
            inputs2 = tokenizer(control_domain_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs1 = model(**inputs1)
                outputs2 = model(**inputs2)
                
                # Extract embeddings for the [CLS] token
                embeddings1 = outputs1.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings2 = outputs2.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Compute cosine similarity between guideline batch and control domain batch
                similarity_scores = cosine_similarity(embeddings1, embeddings2)
            
            # Find the most similar control domain for each guideline in the batch
            for idx1, scores in enumerate(similarity_scores):
                max_score_idx = scores.argmax()
                if scores[max_score_idx] > 0.5:  # Similarity threshold
                    matched_results.append((
                        component_name_batch.iloc[idx1], guideline_batch[idx1], description_batch[idx1],
                        control_domain_batch[max_score_idx], control_statement_batch[max_score_idx], scores[max_score_idx]
                    ))
            
            # Clear memory after processing each batch
            del inputs2, outputs2, embeddings2, similarity_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Clear memory after processing each guideline batch
        del inputs1, outputs1, embeddings1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return matched_results

# Run the batch processing and get the results
results = process_in_batches(guidelines.tolist(), control_domain.tolist(), batch_size=10)

# Create a DataFrame with the matched results, only including the specified columns
output_df = pd.DataFrame(results, columns=[
    'Component Name', 'Guidelines', 'description', 
    'Control domain', 'Standard statement'
])

# Save the output to an Excel file
output_df.to_excel('matched_results.xlsx', index=False)

print("Matching completed and saved to matched_results.xlsx")
