import pandas as pd
from fuzzywuzzy import fuzz

# Load the Excel files
cis_df = pd.read_excel('CIS_Benchmark.xlsx')
master_df = pd.read_excel('master control list.xlsx')

# Extract relevant columns from both DataFrames
cis_relevant = cis_df[['num[age rule', 'description']]
master_relevant = master_df[['control domain', 'standard statement']]

# Define a function to compute similarity score between two texts
def calculate_similarity(text1, text2):
    return fuzz.token_sort_ratio(text1, text2)

# Initialize an empty list to store results
results = []

# Compare each row in CIS_Benchmark.xlsx with each row in master control list.xlsx
for idx_cis, row_cis in cis_relevant.iterrows():
    num_age_rule_cis = row_cis['num[age rule']
    description_cis = row_cis['description']
    
    for idx_master, row_master in master_relevant.iterrows():
        control_domain_master = row_master['control domain']
        standard_statement_master = row_master['standard statement']
        
        # Calculate similarity scores
        num_age_rule_similarity = calculate_similarity(num_age_rule_cis, control_domain_master)
        description_similarity = calculate_similarity(description_cis, standard_statement_master)
        
        # Store the results
        results.append({
            'cis_index': idx_cis,
            'master_index': idx_master,
            'num_age_rule_similarity': num_age_rule_similarity,
            'description_similarity': description_similarity
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Print or save the results
print(results_df)

# Optionally, save results to a new Excel file
results_df.to_excel('comparison_results.xlsx', index=False)
