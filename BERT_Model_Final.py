import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load data
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract necessary columns
guidelines = components_df['Guidelines'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Initialize BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight, fast Sentence-BERT model

# Define batch size
batch_size = 32

# Create a DataFrame to store results
results = []

# Process in batches to handle large datasets
for i in range(0, len(guidelines), batch_size):
    guideline_batch = guidelines[i:i+batch_size]
    guideline_embeddings = model.encode(guideline_batch, convert_to_tensor=True)
    
    for j in range(0, len(control_domains), batch_size):
        control_domain_batch = control_domains[j:j+batch_size]
        control_domain_embeddings = model.encode(control_domain_batch, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_scores = util.pytorch_cos_sim(guideline_embeddings, control_domain_embeddings)

        # Store results
        for k in range(len(guideline_batch)):
            for l in range(len(control_domain_batch)):
                results.append({
                    'Component Name': components_df.iloc[i+k]['Component Name'],
                    'Guidelines': guideline_batch[k],
                    'description': components_df.iloc[i+k]['description'],
                    'Control domain': control_domain_batch[l],
                    'Standard statement': controls_df.iloc[j+l]['Standard statement'],
                    'similarity_score': cosine_scores[k][l].item()
                })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save the results to Excel
results_df.to_excel('Matched_results_Final.xlsx', index=False)
