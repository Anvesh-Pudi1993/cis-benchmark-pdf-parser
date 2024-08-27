import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Excel files
cis_df = pd.read_excel('CIS_Benchmark.xlsx')
master_df = pd.read_excel('master control list.xlsx')

# Extract relevant columns from both DataFrames
cis_relevant = cis_df[['num[age rule', 'description']]
master_relevant = master_df[['control domain', 'standard statement']]

# Flatten the columns for comparison
cis_num_age_rule = cis_relevant['num[age rule'].astype(str)
cis_description = cis_relevant['description'].astype(str)
master_control_domain = master_relevant['control domain'].astype(str)
master_standard_statement = master_relevant['standard statement'].astype(str)

# Combine the columns into single lists for comparison
cis_combined = cis_num_age_rule + ' ' + cis_description
master_combined = master_control_domain + ' ' + master_standard_statement

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data
tfidf_cis = vectorizer.fit_transform(cis_combined)
tfidf_master = vectorizer.transform(master_combined)

# Calculate cosine similarity
similarity_scores = cosine_similarity(tfidf_cis, tfidf_master)

# Convert similarity scores to a DataFrame
similarity_df = pd.DataFrame(similarity_scores,
                             index=cis_relevant.index,
                             columns=master_relevant.index)

# Print the similarity matrix
print(similarity_df)

# Optionally, save the results to a new Excel file
similarity_df.to_excel('cosine_similarity_results.xlsx')
