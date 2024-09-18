import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the two Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # File with component name, guidelines, and description
controls_df = pd.read_excel('MCL.xlsx')   # File with control domain and control statement

# Extract relevant columns
guidelines = components_df['Guidelines'].tolist()
control_statements = controls_df['control statement'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Initialize BERT uncased tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to generate BERT embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():  # Ensure no gradient tracking
        outputs = model(**inputs)
    # Return the mean of token embeddings
    return outputs.last_hidden_state.mean(dim=1)

# Get embeddings for guidelines and control statements
guideline_embeddings = get_embeddings(guidelines).numpy()
control_statement_embeddings = get_embeddings(control_statements).numpy()

# Create an empty list to store results
results = []

# Calculate cosine similarity for each guideline with each control statement
for i, guideline_embedding in enumerate(guideline_embeddings):
    best_score = -1  # Initialize the best score as -1
    best_match = None  # Initialize the best match as None
    
    for j, control_embedding in enumerate(control_statement_embeddings):
        # Compute cosine similarity between guideline and control statement
        similarity_score = cosine_similarity(
            guideline_embedding.reshape(1, -1), 
            control_embedding.reshape(1, -1)
        )[0][0]
        
        # If the current score is the best, update the best match
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = {
                'component name': components_df.iloc[i]['component name'],
                'Guidelines': guidelines[i],
                'description': components_df.iloc[i]['description'],
                'Control domain': control_domains[j],
                'standard statement': control_statements[j],
                'best_similarity_score': best_score
            }
    
    # Add the best match for the current guideline to the results
    results.append(best_match)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a new Excel file
results_df.to_excel('bert_best_similarity_matches.xlsx', index=False)

print("Matching completed and results saved to 'bert_best_similarity_matches.xlsx'")
