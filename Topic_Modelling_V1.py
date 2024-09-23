import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load the two Excel files
components_df = pd.read_excel('CSI_clean_Verify_V1.xlsx')  # Replace with your actual file path
controls_df = pd.read_excel('MCL.xlsx')   # Replace with your actual file path

# Extract relevant columns
guidelines = components_df['Guidelines'].tolist()
descriptions = components_df['description'].tolist()
component_names = components_df['component name'].tolist()
control_statements = controls_df['control statement'].tolist()
control_domains = controls_df['Control domain'].tolist()

# Vectorize guidelines and descriptions separately
guideline_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
description_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Fit and transform guidelines and descriptions into document-term matrices
guideline_dtm = guideline_vectorizer.fit_transform(guidelines)
description_dtm = description_vectorizer.fit_transform(descriptions)

# Train an LDA model for guidelines and descriptions
num_topics = 10  # Adjust the number of topics as needed
guideline_lda = LDA(n_components=num_topics, random_state=42)
description_lda = LDA(n_components=num_topics, random_state=42)

# Fit the LDA models
guideline_lda.fit(guideline_dtm)
description_lda.fit(description_dtm)

# Transform the texts into topic distributions
guideline_topic_distributions = guideline_lda.transform(guideline_dtm)
description_topic_distributions = description_lda.transform(description_dtm)

# Vectorize control domains and control statements separately
control_domain_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
control_statement_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Fit and transform control domains and control statements into document-term matrices
control_domain_dtm = control_domain_vectorizer.fit_transform(control_domains)
control_statement_dtm = control_statement_vectorizer.fit_transform(control_statements)

# Train an LDA model for control domains and statements
control_domain_lda = LDA(n_components=num_topics, random_state=42)
control_statement_lda = LDA(n_components=num_topics, random_state=42)

# Fit the LDA models
control_domain_lda.fit(control_domain_dtm)
control_statement_lda.fit(control_statement_dtm)

# Transform the texts into topic distributions for control domains and statements
control_domain_topic_distributions = control_domain_lda.transform(control_domain_dtm)
control_statement_topic_distributions = control_statement_lda.transform(control_statement_dtm)

# Create a list to store the results
results = []

# Define batch size for processing
batch_size = 32

# Compute similarities between each guideline topic and control statement topic using batches
for i in tqdm(range(0, len(guideline_topic_distributions), batch_size)):
    guideline_batch = guideline_topic_distributions[i:i+batch_size]

    for j in range(0, len(control_statement_topic_distributions), batch_size):
        control_batch = control_statement_topic_distributions[j:j+batch_size]

        # Compute cosine similarity between each guideline and control statement
        similarities = cosine_similarity(guideline_batch, control_batch)

        # Iterate through the similarity matrix and store the best match for each guideline
        for k, guideline_dist in enumerate(guideline_batch):
            best_score = -1
            best_match = None

            for l, control_dist in enumerate(control_batch):
                score = similarities[k][l]

                if score > best_score:
                    best_score = score
                    best_match = {
                        'component name': component_names[i + k],
                        'Guidelines': guidelines[i + k],
                        'description': descriptions[i + k],
                        'Control domain': control_domains[j + l],
                        'control statement': control_statements[j + l],
                        'best_similarity_score': best_score
                    }

            # Append the best match for each guideline
            results.append(best_match)

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# Sort the results by similarity score
results_df = results_df.sort_values(by='best_similarity_score', ascending=False)

# Save the results to an Excel file
results_df.to_excel('best_similarity_matches_topic_model.xlsx', index=False)

print("Matching completed and results saved to 'best_similarity_matches_topic_model.xlsx'")
