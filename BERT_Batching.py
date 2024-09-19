import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the two Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
descriptions = components_df['description'].tolist()
component_names = components_df['component name'].tolist()

control_statements = controls_df['control statement'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings

# Define batch size for processing
batch_size = 32

# Create a list to store the results
results = []

# Process guidelines in batches
for i in range(0, len(guidelines), batch_size):
    guideline_batch = guidelines[i:i+batch_size]
    guideline_embedding_batch = get_embeddings(guideline_batch)

    # Process control statements in batches for each batch of guidelines
    for j in range(0, len(control_statements), batch_size):
        control_batch = control_statements[j:j+batch_size]
        control_embedding_batch = get_embeddings(control_batch)

        # Initialize best match tracker for the current batch of guidelines
        for k, guideline_embedding in enumerate(guideline_embedding_batch):
            best_score = -1  # Best similarity score initialization
            best_match = None  # Initialize best match as None

            # Iterate over the control embeddings in the current batch
            for l, control_embedding in enumerate(control_embedding_batch):
                # Compute cosine similarity between guideline and control statement embeddings
                score = cosine_similarity(
                    guideline_embedding.unsqueeze(0).numpy(),
                    control_embedding.unsqueeze(0).numpy()
                )[0][0]  # Extract the scalar similarity score

                # Update the best match if the current score is higher
                if score > best_score:
                    best_score = score
                    best_match = {
                        'component name': component_names[i+k],
                        'Guidelines': guideline_batch[k],
                        'description': descriptions[i+k],
                        'Control domain': control_domains[j+l],
                        'standard statement': control_batch[l],
                        'best_similarity_score': best_score
                    }

            # Append the best match for the current guideline
            results.append(best_match)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel('best_similarity_matches_batch_bert.xlsx', index=False)

print("Matching completed and results saved to 'best_similarity_matches_batch_bert.xlsx'")
