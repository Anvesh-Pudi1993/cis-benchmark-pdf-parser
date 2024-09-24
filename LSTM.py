import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Load the two Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract relevant columns
guidelines = components_df['Guidelines'].tolist()
descriptions = components_df['description'].tolist()
component_names = components_df['component name'].tolist()
control_statements = controls_df['control statement'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Prepare a combined list of text data
guideline_texts = guidelines
control_texts = control_statements

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = lstm_out[:, -1, :]
        return self.fc(output)

# Tokenizer and Vectorization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_pad(texts, tokenizer, max_length=50):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")

# Create custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=50):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokens = tokenize_and_pad(texts, tokenizer, max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokens['input_ids'][idx], self.tokens['attention_mask'][idx]

# Hyperparameters
embedding_dim = 100
hidden_dim = 128
output_dim = 100
batch_size = 32
epochs = 5
learning_rate = 0.001

# Initialize LSTM Model
vocab_size = tokenizer.vocab_size
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Prepare DataLoader
guideline_dataset = TextDataset(guideline_texts, tokenizer)
control_dataset = TextDataset(control_texts, tokenizer)

guideline_loader = DataLoader(guideline_dataset, batch_size=batch_size, shuffle=False)
control_loader = DataLoader(control_dataset, batch_size=batch_size, shuffle=False)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask in tqdm(guideline_loader):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, outputs)  # Dummy self-reconstruction loss for now
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(guideline_loader)}")

# Get embeddings for guidelines and control statements
def get_embeddings(dataloader, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            output = model(input_ids)
            embeddings.append(output)
    return torch.cat(embeddings, dim=0)

guideline_embeddings = get_embeddings(guideline_loader, model)
control_embeddings = get_embeddings(control_loader, model)

# Compute cosine similarity
cosine_similarities = cosine_similarity(guideline_embeddings.numpy(), control_embeddings.numpy())

# Find the best matches for each guideline
results = []
for i, guideline_embedding in enumerate(guideline_embeddings):
    best_score = -1
    best_match = None
    for j, control_embedding in enumerate(control_embeddings):
        score = cosine_similarities[i][j]
        if score > best_score:
            best_score = score
            best_match = {
                'component name': component_names[i],
                'Guidelines': guidelines[i],
                'description': descriptions[i],
                'Control domain': control_domains[j],
                'standard statement': control_statements[j],
                'best_similarity_score': best_score
            }
    results.append(best_match)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel('best_similarity_matches_lstm.xlsx', index=False)

print("Matching completed and results saved to 'best_similarity_matches_lstm.xlsx'")
