import pandas as pd
import torch
from transformers import AlbertTokenizer, AlbertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the two Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
control_statements = controls_df['control statement'].tolist()

# Initialize ALBERT tokenizer and model
model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertModel.from_pretrained(model_name)

# Function to get ALBERT embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():  # Ensure gradients are not tracked
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Get mean of token embeddings

# Define batch size for processing
batch_size = 32

# Create a list to store the results
results = []

# Process control statements in batches
for i in range(0, len(control_statements), batch_size):
    control_batch = control_statements[i:i+batch_size]
    control_embeddings = get_embeddings(control_batch)

    # Store top 5 best matches for each control statement
    for k in range(len(control_batch)):
        similarity_scores = []

        # Process guidelines in batches
        for j in range(0, len(guidelines), batch_size):
            guideline_batch = guidelines[j:j+batch_size]
            guideline_embeddings = get_embeddings(guideline_batch)

            # Compute cosine similarity between the control statement and each guideline
            for l in range(len(guideline_batch)):
                score = cosine_similarity(
                    control_embeddings[k].unsqueeze(0).numpy(), 
                    guideline_embeddings[l].unsqueeze(0).numpy()
                )[0][0]  # Extract the scalar similarity score

                # Store the similarity score and the relevant data
                similarity_scores.append({
                    'component name': components_df.iloc[j+l]['component name'],
                    'Guidelines': guideline_batch[l],
                    'description': components_df.iloc[j+l]['description'],
                    'Control domain': controls_df.iloc[i+k]['Control domain'],
                    'control statement': control_batch[k],
                    'similarity_score': score
                })

        # Sort the similarity scores for the current control statement and keep the top 5
        top_5_matches = sorted(similarity_scores, key=lambda x: x['similarity_score'], reverse=True)[:5]

        # Append the top 5 matches to the results
        results.extend(top_5_matches)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to an Excel file
results_df.to_excel('top_5_matched_results_with_best_scores_albert.xlsx', index=False)

print("Matching completed and top 5 results saved to 'top_5_matched_results_with_best_scores_albert.xlsx'")
