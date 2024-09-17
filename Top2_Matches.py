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

# Initialize the ALBERT tokenizer and model
model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertModel.from_pretrained(model_name)

# Function to get ALBERT embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean of the token embeddings

# Function to compute top 2 matches for each guideline
def compute_top_matches(guidelines, control_statements, guideline_embeddings, control_statement_embeddings):
    results = []
    
    for g_idx, g_embed in enumerate(guideline_embeddings):
        similarities = []
        for c_idx, c_embed in enumerate(control_statement_embeddings):
            # Compute cosine similarity
            score = cosine_similarity(g_embed.unsqueeze(0).numpy(), c_embed.unsqueeze(0).numpy())[0][0]
            similarities.append({
                'control statement': control_statements[c_idx],
                'Control domain': controls_df.iloc[c_idx]['Control domain'],
                'similarity_score': score
            })
        
        # Sort the similarities and take the top 2
        top_matches = sorted(similarities, key=lambda x: x['similarity_score'], reverse=True)[:2]
        
        for match in top_matches:
            results.append({
                'component name': components_df.iloc[g_idx]['component name'],
                'Guidelines': guidelines[g_idx],
                'description': components_df.iloc[g_idx]['description'],
                'Control domain': match['Control domain'],
                'control statement': match['control statement'],
                'similarity_score': match['similarity_score']
            })
    
    return results

# Get embeddings for both guidelines and control statements
guideline_embeddings = get_embeddings(guidelines)
control_statement_embeddings = get_embeddings(control_statements)

# Compute the top 2 matches for each guideline
results = compute_top_matches(guidelines, control_statements, guideline_embeddings, control_statement_embeddings)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to an Excel file
results_df.to_excel('top2_matches_albert.xlsx', index=False)

print("Matching completed and results saved to 'top2_matches_albert.xlsx'")
