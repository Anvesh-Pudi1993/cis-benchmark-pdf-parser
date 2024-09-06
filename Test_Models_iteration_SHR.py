# %%
import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
import numpy as np

# %%
stig_master_data = pd.read_csv("merged_STIG_data.csv")
stig_master_data.shape

# %%
mcl_data = pd.read_csv("MCL-Gov point.csv",encoding='latin1')
mcl_data.shape

# %%
mcl_data_v2 = mcl_data.copy()

# %%
mcl_data_v2.head()

# %%
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# %%
keywords = ["Weak cryptographic algorithms","Not validated cryptographic algorithms","Encryption","Digital signatures","Data protection","Cryptographic modules","Authentication","Confidentiality","Integrity","Weak cryptography","Attacker","Database","DBMS (Database Management System)","NIST FIPS 140-2","NIST FIPS 140-3","Federal laws","Executive Orders","Directives","Policies","Regulations","Standards","Guidance","NSA Type-X","Hardware-based encryption modules"]

# %%
# keywords_v2 = ["Weak cryptographic algorithms","Digital signatures"]

# %%
keywords_str = " ".join(keywords)
keywords_str

# %%
sentence_encodings = tokenizer.batch_encode_plus(
    mcl_data_v2['Standard Statement'].tolist(), return_tensors='pt', padding=True, truncation=True, add_special_tokens=True
)

# %%
keyword_encodings = tokenizer.batch_encode_plus(
    [keywords_str], return_tensors='pt', padding=True, truncation=True, add_special_tokens=True
)

# %%
input_ids_sentences = sentence_encodings['input_ids']
attention_mask_sentences = sentence_encodings['attention_mask']

# %%
input_ids_keywords = keyword_encodings['input_ids']
attention_mask_keywords = keyword_encodings['attention_mask']


 
# %%
batch_size = 5  
sentence_embeddings_list = []

total_batches = len(input_ids_sentences) // batch_size + (1 if len(input_ids_sentences) % batch_size != 0 else 0)

for i in range(0, len(input_ids_sentences), batch_size):
    
    batch_input_ids = input_ids_sentences[i:i+batch_size]
    batch_attention_mask = attention_mask_sentences[i:i+batch_size]

    
    with torch.no_grad():
        batch_outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        batch_embeddings = batch_outputs.last_hidden_state.mean(dim=1)
        sentence_embeddings_list.append(batch_embeddings)
    
    
    batch_number = (i // batch_size) + 1
    print(f"Completed batch {batch_number}/{total_batches}")


sentence_embeddings = torch.cat(sentence_embeddings_list, dim=0)

# %%
sentence_embeddings

# %%
sentence_embeddings_np = sentence_embeddings.numpy()

# %%
embeddings_list = [embedding.tolist() for embedding in sentence_embeddings]
len(embeddings_list)

# %%
print(embeddings_list)

# %%
df = pd.DataFrame({
    'sentence': mcl_data_v2['Standard Statement'],
    'embedding': embeddings_list
})

# %%
df.to_pickle('sentence_embeddings_and_sentences.pkl')

# %%
# Load the DataFrame
df = pd.read_pickle('sentence_embeddings_and_sentences.pkl')

# Extract sentences and convert embeddings back to PyTorch tensors
loaded_sentences = df['sentence'].tolist()
loaded_embeddings = torch.tensor(df['embedding'].tolist())

# Verify the loaded embeddings
print(len(loaded_embeddings))

# %%
df.head()

# %%
with torch.no_grad():
    keyword_outputs = model(input_ids_keywords, attention_mask=attention_mask_keywords)
    keyword_embedding = keyword_outputs.last_hidden_state.mean(dim=1)

# %%
similarities = cosine_similarity(loaded_embeddings, keyword_embedding)

# %%
mcl_data_v2['Similarity'] = similarities.flatten()

# Sort sentences by similarity score
mcl_data_v2 = mcl_data_v2.sort_values(by='Similarity', ascending=False)

# Display the top matching sentences


# %%
mcl_data_v2

# %%
mcl_data_v2.head()

# %%
#Comparison with the real sentence
example_sentence = """Use of weak or not validated cryptographic algorithms undermines the purposes of utilizing encryption and digital signatures to protect data. Weak algorithms can be easily broken and not validated cryptographic modules may not implement algorithms correctly. Unapproved cryptographic modules or algorithms should not be relied on for authentication, confidentiality, or integrity. Weak cryptography could allow an attacker to gain access to and modify data stored in the database as well as the administration settings of the DBMS.

Applications (including DBMSs) utilizing cryptography are required to use approved NIST FIPS 140-2 or 140-3 validated cryptographic modules that meet the requirements of applicable federal laws, Executive Orders, directives, policies, regulations, standards, and guidance.

NSA Type-X (where X=1, 2, 3, 4) products are NSA-certified, hardware-based encryption modules."""


# %%
example_encodings = tokenizer.batch_encode_plus(
    [example_sentence], return_tensors='pt', padding=True, truncation=True, add_special_tokens=True
)

# %%
example_input_ids = example_encodings['input_ids']
example_attention_mask = example_encodings['attention_mask']

# %%
with torch.no_grad():
    example_outputs = model(example_input_ids, attention_mask=example_attention_mask)
    example_sentence_embedding = example_outputs.last_hidden_state.mean(dim=1)

# %%
similarities_example_sentence = cosine_similarity(loaded_embeddings, example_sentence_embedding)

# %%
mcl_data_v2['Similarity_example_sentence'] = similarities_example_sentence.flatten()
mcl_data_v2['Example_sentence'] = example_sentence

# Sort sentences by similarity score
# mcl_data_v2 = mcl_data_v2.sort_values(by='Similarity', ascending=False)

# %%


# %% [markdown]
# Sentence BERT implementation



# %%
from transformers import RobertaTokenizer, RobertaModel

# %%
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# %%
sentence_encodings = tokenizer.batch_encode_plus(
    mcl_data_v2['Standard Statement'].tolist(), return_tensors='pt', padding=True, truncation=True, add_special_tokens=True
)

# %%
input_ids_sentences = sentence_encodings['input_ids']
attention_mask_sentences = sentence_encodings['attention_mask']

# %%
batch_size = 5  # Adjust batch size according to your system's memory capacity
sentence_embeddings_list = []

total_batches = len(input_ids_sentences) // batch_size + (1 if len(input_ids_sentences) % batch_size != 0 else 0)

for i in range(0, len(input_ids_sentences), batch_size):
    # Slice the batch
    batch_input_ids = input_ids_sentences[i:i+batch_size]
    batch_attention_mask = attention_mask_sentences[i:i+batch_size]

    # Process the batch
    with torch.no_grad():
        batch_outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        # Mean pooling to get sentence embeddings
        batch_embeddings = batch_outputs.last_hidden_state.mean(dim=1)
        sentence_embeddings_list.append(batch_embeddings)
    
    # Print progress
    batch_number = (i // batch_size) + 1
    print(f"Completed batch {batch_number}/{total_batches}")

# Step 4: Concatenate All Batch Embeddings into One Tensor
sentence_embeddings = torch.cat(sentence_embeddings_list, dim=0)


# %%

example_encodings = tokenizer.batch_encode_plus(
    [example_sentence], return_tensors='pt', padding=True, truncation=True, add_special_tokens=True
)

example_input_ids = example_encodings['input_ids']
example_attention_mask = example_encodings['attention_mask']

# %%

with torch.no_grad():
    example_outputs = model(example_input_ids, attention_mask=example_attention_mask)
    example_sentence_embedding = example_outputs.last_hidden_state.mean(dim=1)

# %%
similarities_example_sentence = cosine_similarity(sentence_embeddings, example_sentence_embedding)

# %%
mcl_data_v2['Similarity_example_sentence_RoBERTa'] = similarities_example_sentence.flatten()

# %%
mcl_data_v2.to_csv("MCL_data_with_test_embedding_output.csv", index = False)

# %%
