import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity

# Load Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx',sheet_name=None)   # Replace with your actual file path
controls_df
# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Initialize BERT uncased model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():  # Ensure gradients are not tracked
        outputs = model(**inputs)
    # Use the embeddings from the [CLS] token
    return outputs.last_hidden_state[:, 0, :]

# Define batch size for processing
batch_size = 32

# Create a list to store the results
results = []

# Process guidelines in batches
for i in range(0, len(guidelines), batch_size):
    guideline_batch = guidelines[i:i+batch_size]
    guideline_embeddings = get_embeddings(guideline_batch)

    # Initialize list to store max scores and best matches for the current batch of guidelines
    max_scores = [-1] * len(guideline_batch)
    best_matches = [None] * len(guideline_batch)

    # Process control domains in batches
    for j in range(0, len(control_domains), batch_size):
        control_domain_batch = control_domains[j:j+batch_size]
        control_domain_embeddings = get_embeddings(control_domain_batch)

        # Compute cosine similarity between each guideline and each control domain
        for k in range(len(guideline_batch)):
            for l in range(len(control_domain_batch)):
                # Compute similarity score between the guideline and control domain embeddings
                score = cosine_similarity(
                    guideline_embeddings[k].unsqueeze(0), 
                    control_domain_embeddings[l].unsqueeze(0)
                ).item()  # Extract the scalar similarity score

                # Update the best match for the guideline if the current score is higher
                if score > max_scores[k]:
                    max_scores[k] = score
                    best_matches[k] = {
                        'component name': components_df.iloc[i+k]['component name'],
                        'Guidelines and description': guideline_batch[k],
                        'Control domain': control_domain_batch[l],
                        'control statement': controls_df.iloc[j+l]['control statement'],
                        'best_similarity_score': score
                    }

    # Append the best match for each guideline in the current batch
    results.extend(best_matches)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to an Excel file
results_df.to_excel('matched_results_with_best_scores.xlsx', index=False)

print("Matching completed and results saved to 'matched_results_with_best_scores.xlsx'")
