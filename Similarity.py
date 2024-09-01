import pandas as pd
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df1=pd.read_excel('MCL.xlsx')
df2=pd.read_excel('CSI_clean_Verify_V1.xlsx')
# Correcting the column name for "Control Domain"
df1['Control domain'] = df1['Control domain'].astype(str)

# Extract relevant columns from the CSI file for comparison
csi_combined = df2['Num Page and rule'].astype(str) + " " + df2['description'].astype(str)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer().fit_transform(df1['Control domain'].tolist() + csi_combined.tolist())
vectors = vectorizer.toarray()

# Compute cosine similarity
similarity_matrix = cosine_similarity(vectors[:len(df1['Control domain'])], vectors[len(df1['Control domain']):])

# For each row in the MCL sheet, find the maximum similarity score
max_similarity_scores = similarity_matrix.max(axis=1)

# Add the probability scores to the MCL DataFrame
df2['Probability Score'] = max_similarity_scores

# Save the updated MCL sheet to a new Excel file
output_path = 'Updated_MCL_with_Probabilities.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    df2.to_excel(writer, sheet_name='MCL', index=False)

output_path
