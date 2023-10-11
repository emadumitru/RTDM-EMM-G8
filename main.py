from models import run_sd, data_clean
import pandas as pd

data = pd.read_csv('data/student-mat.csv', sep=';')
data = data_clean(data)
results = run_sd(data, 'G3')

print(f'\n\nNumber of subgroups: {len(results)}\n\n')
print(pd.Series(results.loc[0]))