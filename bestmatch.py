import pandas as pd
from difflib import SequenceMatcher

# Load the Excel files
cis_df = pd.read_excel('/mnt/data/CIS_Benchmark.xlsx')
master_df = pd.read_excel('/mnt/data/master control list.xlsx')

# Extract the relevant columns
num_page_rule = cis_df['Num Page Rule'].dropna()
control_domain = master_df['Control Domain'].dropna()

# Function to calculate similarity
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Compare and filter based on maximum similarity score
results = []

for rule in num_page_rule:
    max_similarity = 0
    best_match = None
    
    for control in control_domain:
        similarity = similar(rule, control)
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = control
    
    if best_match and max_similarity > 0.8:  # You can adjust the threshold
        results.append({
            'Num Page Rule': rule,
            'Most Relevant Control Domain': best_match,
            'Description': cis_df[cis_df['Num Page Rule'] == rule]['Description'].values[0],
            'Standard Statement': master_df[master_df['Control Domain'] == best_match]['Standard Statement'].values[0]
        })
    else:
        results.append({
            'Num Page Rule': rule,
            'Most Relevant Control Domain': None,
            'Description': cis_df[cis_df['Num Page Rule'] == rule]['Description'].values[0],
            'Standard Statement': None
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to Excel
results_df.to_excel('/mnt/data/matched_results.xlsx', index=False)
