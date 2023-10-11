from models import run_sd, data_clean, run_cn2_sd
import pandas as pd


# Load and clean the dataset
data = pd.read_csv('data/student-mat.csv', sep=';')
cleaned_data = data_clean(data)

# Run SD algorithm
results_sd = run_sd(cleaned_data, 'G3')
print(f'\n\nNumber of subgroups (SD): {len(results_sd)}\n\n')
print(pd.Series(results_sd.loc[0]))

# Run CN2-SD algorithm
cn2sd_rules = run_cn2_sd(cleaned_data)
print(f'\n\nNumber of subgroups (CN2-SD): {len(cn2sd_rules)}\n\n')
for rule in cn2sd_rules[:5]:  # Displaying the first 5 rules for review
    print(rule)