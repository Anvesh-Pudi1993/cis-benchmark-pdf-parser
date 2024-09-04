import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity

# Load Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Initialize BERT uncased model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings from the [CLS] token
    return outputs.last_hidden_state[:, 0, :]

# Define batch size
batch_size = 32

# Create a DataFrame to store the results
results = []

# Process in batches
for i in range(0, len(guidelines), batch_size):
    guideline_batch = guidelines[i:i+batch_size]
    guideline_embeddings = get_embeddings(guideline_batch)

    for j in range(0, len(control_domains), batch_size):
        control_domain_batch = control_domains[j:j+batch_size]
        control_domain_embeddings = get_embeddings(control_domain_batch)

        # Compute cosine similarity between each pair of guideline and control domain
        for k in range(len(guideline_batch)):
            max_score = -1  # Initialize maximum similarity score
            best_match = None  # Initialize best match for each guideline

            for l in range(len(control_domain_batch)):
                # Compute similarity score
                score = cosine_similarity(guideline_embeddings[k].unsqueeze(0), control_domain_embeddings[l].unsqueeze(0)).item()
                
                # Update best match if score is higher
                if score > max_score:
                    max_score = score
                    best_match = {
                        'Component Name': components_df.iloc[i+k]['Component Name'],
                        'Guidelines': guideline_batch[k],
                        'description': components_df.iloc[i+k]['description'],
                        'Control domain': control_domain_batch[l],
                        'Standard statement': controls_df.iloc[j+l]['Standard statement'],
                        'best_similarity_score': max_score
                    }

            # Add best match to results
            if best_match:
                results.append(best_match)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to an Excel file
results_df.to_excel('matched_results_with_best_scores.xlsx', index=False)
