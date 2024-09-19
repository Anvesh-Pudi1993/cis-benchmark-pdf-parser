import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # File path for your components Excel
controls_df = pd.read_excel('MCL.xlsx')   # File path for your control statements Excel

# Extract relevant columns
guidelines = components_df['Guidelines'].tolist()
descriptions = components_df['description'].tolist()
component_names = components_df['component name'].tolist()

control_statements = controls_df['control statement'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Combine guidelines and control statements into a single list
combined_text = guidelines + control_statements

# Fit the vectorizer on combined text
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)

# Split the TF-IDF matrix into guidelines and control statements
guideline_tfidf = tfidf_matrix[:len(guidelines)]
control_tfidf = tfidf_matrix[len(guidelines):]

# Create a list to store results
results = []

# Compute cosine similarity between each guideline and control statement
for i, guideline_vector in enumerate(guideline_tfidf):
    best_score = -1  # Initialize the best score as -1
    best_match = None  # Initialize the best match as None

    for j, control_vector in enumerate(control_tfidf):
        # Compute cosine similarity between guideline and control statement
        similarity_score = cosine_similarity(guideline_vector, control_vector)[0][0]

        # Update best match if the current similarity score is higher
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = {
                'component name': component_names[i],
                'Guidelines': guidelines[i],
                'description': descriptions[i],
                'Control domain': control_domains[j],
                'standard statement': control_statements[j],
                'best_similarity_score': best_score
            }

    # Append the best match for the current guideline to the results
    results.append(best_match)

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel('tfidf_best_similarity_matches.xlsx', index=False)

print("Matching completed and results saved to 'tfidf_best_similarity_matches.xlsx'")
