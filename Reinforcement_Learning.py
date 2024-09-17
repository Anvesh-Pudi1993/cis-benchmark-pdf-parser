import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the two Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
control_statements = controls_df['control statement'].tolist()

# Initialize a BERT tokenizer and model (could be any RL-compatible model for embeddings)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean of the token embeddings

# Get embeddings for guidelines and control statements
guideline_embeddings = get_embeddings(guidelines)
control_statement_embeddings = get_embeddings(control_statements)

# Q-learning parameters
learning_rate = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

# Initialize Q-table (guideline_index, control_statement_index)
q_table = np.zeros((len(guidelines), len(control_statements)))

# Reward function (cosine similarity)
def calculate_reward(embedding1, embedding2):
    return cosine_similarity(embedding1.unsqueeze(0).numpy(), embedding2.unsqueeze(0).numpy())[0][0]

# Q-learning function
for episode in range(num_episodes):
    for guideline_idx in range(len(guideline_embeddings)):
        # Epsilon-greedy selection
        if np.random.uniform(0, 1) < epsilon:
            # Random action (control statement selection)
            action = np.random.choice(len(control_statements))
        else:
            # Best action from Q-table
            action = np.argmax(q_table[guideline_idx])

        # Get the current guideline and selected control statement embeddings
        guideline_embedding = guideline_embeddings[guideline_idx]
        control_statement_embedding = control_statement_embeddings[action]

        # Calculate reward based on similarity
        reward = calculate_reward(guideline_embedding, control_statement_embedding)

        # Find the max future Q-value
        max_future_q = np.max(q_table[guideline_idx])

        # Update Q-value using the Q-learning formula
        q_table[guideline_idx, action] = q_table[guideline_idx, action] + learning_rate * (reward + gamma * max_future_q - q_table[guideline_idx, action])

# Find the best matches based on the final Q-table
best_matches = []
for guideline_idx in range(len(guideline_embeddings)):
    best_control_idx = np.argmax(q_table[guideline_idx])
    best_matches.append({
        'component name': components_df.iloc[guideline_idx]['component name'],
        'Guidelines': guidelines[guideline_idx],
        'description': components_df.iloc[guideline_idx]['description'],
        'Control domain': controls_df.iloc[best_control_idx]['Control domain'],
        'control statement': controls_df.iloc[best_control_idx]['control statement'],
        'best_similarity_score': calculate_reward(guideline_embeddings[guideline_idx], control_statement_embeddings[best_control_idx])
    })

# Convert results to DataFrame
results_df = pd.DataFrame(best_matches)

# Save results to an Excel file
results_df.to_excel('rl_matched_results.xlsx', index=False)

print("Matching completed and results saved to 'rl_matched_results.xlsx'")
