import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import Orange
import pysubgroup as ps

def data_clean(data):
    data = data.dropna()
    data['school'] = data['school'].map({'GP': 0, 'MS': 1})
    data['sex'] = data['sex'].map({'F': 0, 'M': 1})
    data['address'] = data['address'].map({'U': 0, 'R': 1})
    data['famsize'] = data['famsize'].map({'LE3': 0, 'GT3': 1})
    data['Pstatus'] = data['Pstatus'].map({'T': 0, 'A': 1})
    data['schoolsup'] = data['schoolsup'].map({'no': 0, 'yes': 1})
    data['famsup'] = data['famsup'].map({'no': 0, 'yes': 1})
    data['paid'] = data['paid'].map({'no': 0, 'yes': 1})
    data['activities'] = data['activities'].map({'no': 0, 'yes': 1})
    data['nursery'] = data['nursery'].map({'no': 0, 'yes': 1})
    data['higher'] = data['higher'].map({'no': 0, 'yes': 1})
    data['internet'] = data['internet'].map({'no': 0, 'yes': 1})
    data['romantic'] = data['romantic'].map({'no': 0, 'yes': 1})
    data = pd.get_dummies(data, columns=['Mjob', 'Fjob', 'reason', 'guardian'])
    data = data.drop(['G1', 'G2'], axis=1)
    scaler = MinMaxScaler()
    data[['age', 'absences']] = scaler.fit_transform(data[['age', 'absences']])
    return data

def run_sd(data, tg):
    target = ps.NumericTarget(tg)  
    search_space = ps.create_selectors(data, ignore=[tg])
    qf = ps.StandardQFNumeric(1.0)  
    task = ps.SubgroupDiscoveryTask(data, target, search_space, qf)
    result = ps.SimpleSearch().execute(task)
    result_df2 = result.to_dataframe()
    return(result_df2)


def run_cn2_sd(data):
    # Convert data to Orange data table
    domain = Orange.data.Domain([Orange.data.ContinuousVariable.make(name) for name in data.columns])
    data_table = Orange.data.Table.from_numpy(domain, data.values)

    # Discretize the 'G3' variable
    discretizer = Orange.preprocess.Discretize()
    discretizer.method = Orange.preprocess.discretize.EqualFreq(n=5)
    data_table_discretized = discretizer(data_table)

    # Ensure unique variable names
    unique_vars = set(data_table_discretized.domain.variables)
    unique_vars.discard(data_table_discretized.domain["G3"])

    # Update the domain to make 'G3' as a class variable
    new_domain = Orange.data.Domain(unique_vars, data_table_discretized.domain["G3"])
    data_table_discretized = Orange.data.Table.from_table(new_domain, data_table_discretized)

    # Set up the CN2-SD learner for subgroup discovery
    cn2sd_learner = Orange.classification.rules.CN2SDLearner()

    # Perform subgroup discovery
    cn2sd_classifier = cn2sd_learner(data_table_discretized)

    # Extract and return discovered subgroups (rules)
    return cn2sd_classifier.rule_list



