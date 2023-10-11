import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import Orange
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

# def run_orange(): 
    data_table = Orange.data.Table.from_numpy(None, data.values, data.index)
    # from orangecontrib.associate.fpgrowth import *

    # itemsets = list(frequent_itemsets(data_table, min_support=0.1))  # Adjust support as needed
    # itemsets_dict = {frozenset(itemset): support for itemset, support in itemsets}
    # rules = association_rules(itemsets, min_confidence=0.5)  
    # print(rules)
    # subgroups = []
    # for rule in rules:
    #     antecedent = rule[0]
    #     if antecedent:  # Ensure the antecedent (conditions) exist
    #         subgroup = {item[0] for item in antecedent}
    #         subgroups.append(subgroup)\


    from Orange.classification import CN2Learner

    data['G3'] = pd.cut(data['G3'], bins=5) 

    data_table = Orange.data.Table(data.values)

    # Set up the CN2-SD learner for subgroup discovery
    cn2sd_learner = Orange.classification.rules.CN2SDLearner()

    # Specify the target variable
    target_variable = Orange.data.Variable(data_table.domain, 'G3')
    cn2sd_learner.rule_finder.target_class = target_variable

    # Perform subgroup discovery
    cn2sd_classifier = cn2sd_learner(data_table)

    # Extract and display discovered subgroups (rules)
    for rule in cn2sd_classifier.rule_list:
        print(rule)

