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

# Compare and find the best match for each rule
results = []

for rule in num_page_rule:
    max_similarity = 0
    best_match = None
    
    for control in control_domain:
        similarity = similar(rule, control)
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = control
    
    # Find the corresponding description, standard statement, component name, and the guideline itself
    description = cis_df[cis_df['Num Page Rule'] == rule]['Description'].values[0]
    guideline = cis_df[cis_df['Num Page Rule'] == rule]['Guideline'].values[0]  # Assuming 'Guideline' is the column name
    standard_statement = master_df[master_df['Control Domain'] == best_match]['Standard Statement'].values[0] if best_match else None
    component_name = master_df[master_df['Control Domain'] == best_match]['Component Name'].values[0] if best_match else None

    # Append the result to the list
    results.append({
        'Num Page Rule': rule,
        'Corresponding Guideline': guideline,
        'Most Relevant Control Domain': best_match,
        'Description': description,
        'Standard Statement': standard_statement,
        'Component Name': component_name
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to Excel
results_df.to_excel('/mnt/data/matched_results.xlsx', index=False)
