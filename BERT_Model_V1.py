import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the Excel files
csi_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')
mcl_df = pd.read_excel('MCL.xlsx')

# Extract relevant columns
csi_descriptions = csi_df['description'].fillna("").tolist()
mcl_control_domains = mcl_df['Control domain'].fillna("").tolist()

# Load pre-trained BERT model from Sentence-Transformers
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Encode the sentences
csi_embeddings = model.encode(csi_descriptions, convert_to_tensor=True)
mcl_embeddings = model.encode(mcl_control_domains, convert_to_tensor=True)

# Calculate cosine similarities
similarities = util.cos_sim(csi_embeddings, mcl_embeddings)

# Find the most similar MCL control domain for each CSI description
most_similar_mcl_idx = similarities.argmax(dim=1).cpu().numpy()
most_similar_scores = similarities.max(dim=1).values.cpu().numpy()

# Prepare the results
results = pd.DataFrame({
    "CSI_Description": csi_descriptions,
    "Most_Similar_MCL_Control_Domain": [mcl_control_domains[i] for i in most_similar_mcl_idx],
    "Similarity_Score": most_similar_scores
})

# Show the top results
print(results.head())

# Optionally, save results to an Excel file
results.to_excel("Comparison_Results.xlsx", index=False)
