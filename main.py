from models import sd, data_clean, cn2_sd, apriori_sd, sd_map, dssd, nmeef_sd
import pandas as pd


# Load and clean the dataset
data = pd.read_csv('data/student-mat.csv', sep=';')
cleaned_data = data_clean(data)

# Run SD algorithm
results_sd = sd(cleaned_data, target_column='G3')
print(f'\n\nNumber of subgroups (SD): {len(results_sd)}\n\n')
print(results_sd.head())  # Displaying the first few rules for review

# Run CN2-SD algorithm
cn2sd_results = cn2_sd(cleaned_data, target_column='G3')
print(f'\n\nNumber of subgroups (CN2-SD): {len(cn2sd_results)}\n\n')
print(cn2sd_results.head())

# Run the SD-Map algorithm
sd_map_results = sd_map(cleaned_data, target_column='G3', min_support=0.1)
print(f'\n\nNumber of subgroups (SD-Map): {len(sd_map_results)}\n\n')
print(sd_map_results.head())  # Displaying the first few rules for review

# Run the DSSD algorithm
dssd_results = dssd(cleaned_data, target_column='G3', min_support=0.1)
print(f'\n\nNumber of subgroups (DSSD): {len(dssd_results)}\n\n')
print(dssd_results.head())  # Displaying the first few rules for review

# Run the NMEEF-SD algorithm
nmeef_sd_results = nmeef_sd(data, target_column='G3', n_generations=50, population_size=100)
print(f'\n\nNumber of subgroups (NMEEF-SD): {len(nmeef_sd_results)}\n\n')
print(nmeef_sd_results.head())  # Displaying the first few rules for review

# Run the APRIORI-SD algorithm
apriori_sd_results = apriori_sd(cleaned_data, target_column='G3', min_support=0.1, metric="lift", min_threshold=0.1)
print(f'\n\nNumber of subgroups (APRIORI-SD): {len(apriori_sd_results)}\n\n')
print(apriori_sd_results.head())  # Displaying the first few rules for review