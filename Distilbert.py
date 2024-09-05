import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the data from Excel files
components_df = pd.read_excel('path_to_first_excel.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('path_to_second_excel.xlsx')   # Replace with your actual file path

# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Initialize the DistilBERT uncased model
model = SentenceTransformer('distilbert-base-uncased')

# Define batch size for processing
batch_size = 32

# Create a list to store the results
results = []

# Process "Guidelines" in batches
for i in range(0, len(guidelines), batch_size):
    guideline_batch = guidelines[i:i+batch_size]
    guideline_embeddings = model.encode(guideline_batch, convert_to_tensor=True)
    
    # Process "Control domains" in batches
    for j in range(0, len(control_domains), batch_size):
        control_domain_batch = control_domains[j:j+batch_size]
        control_domain_embeddings = model.encode(control_domain_batch, convert_to_tensor=True)

        # Compute cosine similarity between each pair of guideline and control domain
        cosine_scores = util.pytorch_cos_sim(guideline_embeddings, control_domain_embeddings)

        # For each guideline, find the best matching control domain
        for k in range(len(guideline_batch)):
            max_score = -1  # Initialize maximum similarity score
            best_match = None  # Initialize best match for each guideline
            
            # Iterate through control domains and find the best match
            for l in range(len(control_domain_batch)):
                score = cosine_scores[k][l].item()
                
                # Update best match if score is higher
                if score > max_score:
                    max_score = score
                    best_match = {
                        'component name': components_df.iloc[i+k]['component name'],
                        'Guidelines': guideline_batch[k],
                        'description': components_df.iloc[i+k]['description'],
                        'Control domain': control_domain_batch[l],
                        'control statement': controls_df.iloc[j+l]['control statement'],
                        'best_similarity_score': max_score
                    }
            
            # Add the best match to results
            if best_match:
                results.append(best_match)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel('matched_results_with_best_scores.xlsx', index=False)

print("Matching completed and results saved to 'matched_results_with_best_scores.xlsx'")
