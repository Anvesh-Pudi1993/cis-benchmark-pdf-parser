import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.nn.functional import cosine_similarity

# Load Excel files
components_df = pd.read_excel('CSI_clean_Test.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('mcl_test.csv',engine='openpyxl')   # Replace with your actual file path
# controls_df=controls_df['mcl_test']
# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
control_domains = controls_df['Control Domain'].tolist() 

# Initialize RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# Function to get RoBERTa embeddings
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
for i in range(len(control_domains), batch_size):
    control_domain_batch = control_domains[i:i+batch_size]
    control_domain_embeddings = get_embeddings(control_domain_batch)

    # Initialize list to store max scores and best matches for the current batch of guidelines
    max_scores = [-1] * len(control_domain_batch)
    best_matches = [None] * len(control_domain_batch)

    # Process control domains in batches
    for j in range(0, len(guidelines), batch_size):
        guideline_batch = guidelines[j:j+batch_size]
        guideline_embeddings = get_embeddings(guideline_batch)

        # Compute cosine similarity between each guideline and each control domain
        for k in range(len(control_domain_batch)):
            for l in range(len(guideline_batch)):
                # Compute similarity score between the guideline and control domain embeddings
                score = cosine_similarity(control_domain_embeddings[k].unsqueeze(0),guideline_embeddings[l].unsqueeze(0)).item()  # Extract the scalar similarity score

                # Update the best match for the guideline if the current score is higher
                if score > max_scores[l]:
                    max_scores[l] = score
                    best_matches[l] = {
                        'Component Name': components_df.iloc[j+l]['Component Name'],
                        'Guidelines': guideline_batch[j],
                        'Control domain': control_domain_batch[k],
                        'Standard Statement': controls_df.iloc[j+l]['Standard Statement'],
                        'best_similarity_score': score
                    }

    # Append the best match for each guideline in the current batch
    results.extend(best_matches)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to an Excel file
results_df.to_excel('matched_results_with_best_scores_Roberta.xlsx', index=False)

print("Matching completed and results saved to 'matched_results_with_best_scores_Roberta.xlsx'")
