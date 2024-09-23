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

# Combine guidelines and descriptions into a single text field
guideline_texts = [f"{guideline} {desc}" for guideline, desc in zip(guidelines, descriptions)]
control_texts = [f"{domain} {statement}" for domain, statement in zip(control_domains, control_statements)]

# Combine both lists of texts for topic modeling
all_texts = guideline_texts + control_texts

# Vectorize the text using CountVectorizer (Bag-of-Words)
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
text_vectors = vectorizer.fit_transform(all_texts)

# Train an LDA model to find latent topics
num_topics = 10  # You can adjust the number of topics based on your dataset
lda = LDA(n_components=num_topics, random_state=42)
lda.fit(text_vectors)

# Transform the texts into topic distributions
text_topic_distributions = lda.transform(text_vectors)

# Split the topic distributions into guideline and control parts
guideline_topic_distributions = text_topic_distributions[:len(guideline_texts)]
control_topic_distributions = text_topic_distributions[len(guideline_texts):]

# Create a list to store the results
results = []

# Define batch size for processing
batch_size = 32

# Compute similarities between each guideline and control statement using batches
for i in tqdm(range(0, len(guideline_topic_distributions), batch_size)):
    guideline_batch = guideline_topic_distributions[i:i+batch_size]

    for j in range(0, len(control_topic_distributions), batch_size):
        control_batch = control_topic_distributions[j:j+batch_size]

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
                        'component name': component_names[i+k],
                        'Guidelines and description': guideline_texts[i+k],
                        'Control domain': control_domains[j+l],
                        'control statement': control_statements[j+l],
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
