import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the two Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
control_statements = controls_df['control statement'].tolist()

# Initialize BERT tokenizer and model for embedding extraction
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to get embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean of token embeddings

# Get embeddings for guidelines and control statements
guideline_embeddings = get_embeddings(guidelines).numpy()
control_statement_embeddings = get_embeddings(control_statements).numpy()

# Prepare the training dataset by calculating cosine similarity between guideline and control statement embeddings
similarity_scores = cosine_similarity(guideline_embeddings, control_statement_embeddings)

# Create a labeled dataset (positive pairs where similarity score > threshold)
X = []
y = []

# We treat similarity scores > 0.8 as positive examples and less as negative examples
threshold = 0.8
for i in range(len(guidelines)):
    for j in range(len(control_statements)):
        X.append(np.concatenate((guideline_embeddings[i], control_statement_embeddings[j])))
        y.append(1 if similarity_scores[i, j] > threshold else 0)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities for the test set
y_proba = clf.predict_proba(X_test)[:, 1]

# Predict probabilities for all pairs of guidelines and control statements
all_proba = clf.predict_proba(scaler.transform(X))[:, 1]

# Get top 2 matches for each guideline
top_2_matches = []
for i in range(len(guidelines)):
    # Get the similarity scores for guideline i with all control statements
    scores = all_proba[i * len(control_statements): (i + 1) * len(control_statements)]
    
    # Get the top 2 control statements based on similarity score
    top_indices = np.argsort(scores)[-2:][::-1]
    
    for idx in top_indices:
        top_2_matches.append({
            'component name': components_df.iloc[i]['component name'],
            'Guidelines': guidelines[i],
            'description': components_df.iloc[i]['description'],
            'Control domain': controls_df.iloc[idx]['Control domain'],
            'standard statement': control_statements[idx],
            'similarity_score': scores[idx]
        })

# Convert results to DataFrame
results_df = pd.DataFrame(top_2_matches)

# Save results to an Excel file
results_df.to_excel('top_2_matches_classifier.xlsx', index=False)

print("Matching completed and results saved to 'top_2_matches_classifier.xlsx'")
