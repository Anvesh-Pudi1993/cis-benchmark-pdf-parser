import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

# Load the two Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract the relevant columns
guidelines = components_df['Guidelines'].tolist()
descriptions = components_df['description'].tolist()
component_names = components_df['component name'].tolist()

control_statements = controls_df['control statement'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Tokenizer and Pretrained BERT Model (for fine-tuning)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Function to tokenize sentence pairs for the BERT model
def tokenize_pairs(guideline, control_statement, max_length=128):
    return tokenizer(guideline, control_statement, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

# Custom Dataset Class for the pairs
class SentencePairDataset(Dataset):
    def __init__(self, guidelines, control_statements, tokenizer, max_length=128):
        self.guidelines = guidelines
        self.control_statements = control_statements
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.guidelines) * len(self.control_statements)

    def __getitem__(self, idx):
        guideline_idx = idx // len(self.control_statements)
        control_idx = idx % len(self.control_statements)
        guideline = self.guidelines[guideline_idx]
        control_statement = self.control_statements[control_idx]
        
        inputs = tokenize_pairs(guideline, control_statement, self.max_length)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'token_type_ids': inputs['token_type_ids'].squeeze(),
            'guideline_idx': guideline_idx,
            'control_idx': control_idx
        }

# Load data into dataset
dataset = SentencePairDataset(guidelines, control_statements, tokenizer)
batch_size = 16  # Adjust batch size as needed

# DataLoader to batch the inputs
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Set up training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training function (you can customize epochs, and save the model after training)
def train_model(model, dataloader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            # Fake labels for similarity (can modify based on actual similarity or your task)
            labels = torch.zeros(len(input_ids), dtype=torch.long).to(device)  # Binary labels: 0 or 1

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Fine-tune the BERT model on your data
train_model(model, dataloader)

# Now we use the fine-tuned BERT to generate similarity scores
model.eval()

# Function to calculate similarity scores between guideline and control statement embeddings
def calculate_similarity(model, dataloader):
    similarity_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            outputs = model.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # Compute cosine similarity between the embeddings
            for i, embedding in enumerate(embeddings):
                guideline_idx = batch['guideline_idx'][i].item()
                control_idx = batch['control_idx'][i].item()
                similarity = cosine_similarity(embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
                similarity_scores.append((guideline_idx, control_idx, similarity))
    
    return similarity_scores

# Get similarity scores between guideline and control statements
similarity_scores = calculate_similarity(model, dataloader)

# Convert similarity scores to DataFrame and get the best match for each guideline
results = []
for guideline_idx, control_idx, score in similarity_scores:
    results.append({
        'component name': component_names[guideline_idx],
        'Guidelines': guidelines[guideline_idx],
        'description': descriptions[guideline_idx],
        'Control domain': control_domains[control_idx],
        'standard statement': control_statements[control_idx],
        'similarity_score': score
    })

# Create DataFrame for the results and save to Excel
results_df = pd.DataFrame(results)

# Sort by similarity score for each guideline and get the best match
results_df = results_df.sort_values(by='similarity_score', ascending=False).groupby('Guidelines').head(1)

# Save the results
results_df.to_excel('best_similarity_matches_finetuned_bert.xlsx', index=False)

print("Matching completed and results saved to 'best_similarity_matches_finetuned_bert.xlsx'")
