import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import Orange
from scipy.stats import ranksums
from Orange.data import Table, Domain, ContinuousVariable
from Orange.classification import CN2Learner
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor

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


def sd(data, target_column):
    """
    Subgroup Discovery (SD) using a decision tree.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.

    Returns:
    - pd.DataFrame containing rules (as interpretable conditions) and their quality measures.
    """

    # Train a shallow decision tree
    tree = DecisionTreeRegressor(max_depth=2)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    tree.fit(X, y)

    # Extract rules from the tree
    rules = []
    path = tree.decision_path(X).todense()

    for sample in range(path.shape[0]):
        rule = []
        for node in range(path.shape[1]):
            if path[sample, node] == 0:
                continue
            if (tree.tree_.children_left[node] == tree.tree_.children_right[node]):
                continue
            if X.iloc[sample, tree.tree_.feature[node]] <= tree.tree_.threshold[node]:
                rule.append(f"{X.columns[tree.tree_.feature[node]]} <= {tree.tree_.threshold[node]:.2f}")
            else:
                rule.append(f"{X.columns[tree.tree_.feature[node]]} > {tree.tree_.threshold[node]:.2f}")
        rules.append(' AND '.join(rule))

    # Calculate quality for each rule (difference from global mean)
    overall_mean = y.mean()
    qualities = [y[i] - overall_mean for i in range(len(y))]

    coverage_list = [0] * len(rules)
    support_list = [0] * len(rules)

    for i in range(len(rules)):
        rule = rules[i]
        subgroup_data = data.query(rule)
        coverage = len(subgroup_data)
        support = coverage / len(data)
        coverage_list[i] = coverage
        support_list[i] = support

    # Number of subgroups
    num_subgroups = len(rules)

    # Average length of subgroups
    avg_length_subgroups = np.mean([len(rule.split(" AND ")) for rule in rules])

    # Convert rules and metrics into DataFrame
    rules_df = pd.DataFrame({
        'rule': rules,
        'quality': qualities,
        'coverage': coverage_list,
        'support': support_list
    })

    result_metrics = {
        'Average Quality': np.mean(rules_df.quality),
        'Average Coverage': np.mean(coverage_list),
        'Average Support': np.mean(support_list),
        'Number of Subgroups': num_subgroups,
        'Average Length of Subgroups': avg_length_subgroups,
    }

    return result_metrics


def cn2_sd(data, target_column):
    """
    Subgroup Discovery using CN2 algorithm with adjusted parameters.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.

    Returns:
    - pd.DataFrame containing rules (as interpretable conditions) and their quality measures.
    """

    # Convert all columns to string type for categorical interpretation
    data_str = data.astype(str)

    def get_subgroup(rule, data):
        condition_str = str(rule).split('IF ')[1].split(' THEN')[0].strip()
        conditions = condition_str.split(' AND ')

        # Apply each condition to filter the subgroup
        subgroup = data
        for condition in conditions:
            # Extract the column name, comparison operator, and value
            if "==" in condition:
                col_name, value = condition.split('==')
                subgroup = subgroup[subgroup[col_name] == float(value)]
            elif "!=" in condition:
                col_name, value = condition.split('!=')
                subgroup = subgroup[subgroup[col_name] == float(value)]
        return subgroup

    # Create Orange domain with explicit categories for each column
    domain_vars = []
    for col in data_str.columns:
        unique_vals = data_str[col].unique()
        domain_vars.append(Orange.data.DiscreteVariable.make(col, values=unique_vals))
    domain = Domain(domain_vars[:-1], domain_vars[-1])

    table = Table.from_list(domain, data_str.values)

    # Create learner with adjusted parameters and induce rules
    learner = CN2Learner()
    learner.rule_finder.search_algorithm.beam_width = 10
    learner.rule_finder.general_validator.min_covered_examples = 15
    classifier = learner(table)

    rules = []
    for rule in classifier.rule_list:
        # Using string representation to extract the rule conditions
        rule_str = str(rule).split("->")[0].strip()
        subgroup = get_subgroup(rule, data)
        coverage = len(subgroup) / len(data)

        TP = 0
        for i in list(subgroup.index)[1:]:
            if (subgroup[target_column][i] == 1) & (data[target_column][i] == 1):
                TP += 1

        # true_positives = sum([data[target_column][idx] == 1 for idx in range(len(coverage))])
        FP = len(subgroup) - TP
        # true_negatives = len(data) - len(subgroup) - FP

        support = TP / len(data)
        quality = rule.quality

        # wracc = (len(subgroup) / len(data)) * (((true_positives + true_negatives) / len(subgroup)) - (len(data[data[target_column] == 1]) / len(data)))
        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))


        # Calculate Significance using likelihood ratio statistic
        non_subgroup_data = data[~data.index.isin(subgroup.index)].dropna()
        subgroup = subgroup.dropna()
        # subgroup = subgroup[target_column].astype(int)
        # non_subgroup_data = non_subgroup_data[target_column].astype(int)
        # print(subgroup[target_column])
        # print(non_subgroup_data.iloc[:,0])
        _, significance = ranksums(subgroup[target_column], non_subgroup_data.iloc[:,0])
        # significance = 2 * true_positives * np.log(true_positives / (len(data[target_column]) * (len(subgroup)/len(data))))

        # Calculate Confidence
        confidence = (TP / len(subgroup)) if len(subgroup) > 0 else 0

        rules.append((rule_str, quality, coverage, support, wracc, significance, confidence))

    # Convert rules into DataFrame
    rules_df = pd.DataFrame(rules, columns=['rule', 'quality', 'coverage', 'support', 'WRAcc', 'Significance', 'Confidence'])

    num_subgroups = len(rules)
    av_len_subgroups = sum(len(rule[0].split(' AND ')) for rule in rules) / num_subgroups

    result_metrics = {
        'Average Quality': np.mean(rules_df.quality),
        'Average Coverage': np.mean(rules_df.coverage),
        'Average Support': np.mean(rules_df.support),
        'Average WRAcc': np.mean(rules_df.WRAcc),
        'Average Significance': np.mean(rules_df.Significance),
        'Average Confidence': np.mean(rules_df.Confidence),
        'Number of Subgroups': num_subgroups,
        'Average Length of Subgroups': av_len_subgroups,
    } 

    # return rules_df.sort_values(by='quality', ascending=False), num_subgroups, len_subgroups
    return result_metrics


def sd_map(data, target_column, min_support):
    """
    SD-Map algorithm for subgroup discovery with binarized input.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.
    - min_support: float, minimum support for the Apriori algorithm.

    Returns:
    - pd.DataFrame containing subgroups and their quality measures.
    """

    # Binarize the input data based on median values
    for column in data.columns:
        if column != target_column:
            median_value = data[column].median()
            data[column] = (data[column] > median_value).astype(int)

    # Drop the target column and compute frequent itemsets using Apriori
    frequent_itemsets = apriori(data.drop(columns=[target_column]), min_support=min_support, use_colnames=True)

    # Compute the quality of each subgroup
    overall_mean = data[target_column].mean()
    quality_measures = []
    coverage_list = []
    support_list = []
    wracc_list = []
    significance_list = []
    confidence_list = []
    len_list = []

    for _, row in frequent_itemsets.iterrows():
        subgroup_data = data[np.logical_and.reduce([data[col] for col in row['itemsets']])]
        # print(subgroup_data)
        subgroup_mean = subgroup_data[target_column].mean()
        coverage = len(subgroup_data) / len(data)
        len_rule = len(row['itemsets'])
        
        
        # Calculate Wilcoxon Rank Sum Test (Significance)
        non_subgroup_data = data[~np.logical_and.reduce([data[col] for col in row['itemsets']])]
        _, p_value = ranksums(subgroup_data[target_column], non_subgroup_data[target_column])
        
        # Calculate TP, TN, FP, FN for the subgroup
        TP = len(subgroup_data[subgroup_data[target_column] == 1])
        FP = len(subgroup_data[subgroup_data[target_column] == 0])

        support = TP / len(data)

        # Calculate Confidence
        confidence = (TP / len(subgroup_data)) if len(subgroup_data) > 0 else 0

        # Calculate WRAcc
        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))
        # wracc = (len(subgroup) / len(data)) * (((true_positives + true_negatives) / len(subgroup)) - (len(data[data[target_column] == 1]) / len(data)))


        quality_measures.append(subgroup_mean - overall_mean)
        coverage_list.append(coverage)
        support_list.append(support)
        wracc_list.append(wracc)
        significance_list.append(p_value)
        confidence_list.append(confidence)
        len_list.append(len_rule)

    frequent_itemsets['quality'] = quality_measures
    frequent_itemsets['WRacc'] = wracc_list
    frequent_itemsets['Significance'] = significance_list
    frequent_itemsets['Confidence'] = confidence_list

    # Rank subgroups based on quality
    ranked_subgroups = frequent_itemsets.sort_values(by='quality', ascending=False)

    result_metrics = {
        'Average Quality': np.mean(quality_measures),
        'Average Coverage': np.mean(coverage_list),
        'Average Support': np.mean(support_list),
        'Average WRacc': np.mean(wracc_list),
        'Average Significance': np.mean(significance_list),
        'Average Confidence': np.mean(confidence_list),
        'Number of Subgroups': len(frequent_itemsets),
        'Average Length of Subgroups': np.mean(len_list),
    }

    return result_metrics


def dssd(data, target_column, min_support):
    """
    Direct Subgroup Set Discovery (DSSD) algorithm for subgroup discovery with binarized input.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.
    - min_support: float, minimum support for the Apriori algorithm.

    Returns:
    - pd.DataFrame containing non-redundant subgroups and their quality measures.
    """

    # Binarize the input data based on median values
    for column in data.columns:
        if column != target_column:
            median_value = data[column].median()
            data[column] = (data[column] > median_value).astype(int)

    # Compute frequent itemsets using Apriori
    frequent_itemsets = apriori(data.drop(columns=[target_column]), min_support=min_support, use_colnames=True)

    # Compute the quality of each subgroup
    overall_mean = data[target_column].mean()

    quality_measures = []
    coverage_list = []
    support_list = []
    wracc_list = []
    significance_list = []
    confidence_list = []

    # print(frequent_itemsets["itemsets"])

    for _, row in frequent_itemsets.iterrows():
        subgroup_data = data[np.logical_and.reduce([data[col] for col in row['itemsets']])]

        subgroup_mean = subgroup_data[target_column].mean()
        quality = subgroup_mean - overall_mean

        # Calculate coverage
        coverage = len(subgroup_data) / len(data)

        # Calculate Significance
        non_subgroup_data = data[~data.index.isin(subgroup_data.index)]
        _, p_value = ranksums(subgroup_data[target_column], non_subgroup_data[target_column])

        TP = len(subgroup_data[subgroup_data[target_column] == 1])
        FP = len(subgroup_data[subgroup_data[target_column] == 0])

        # Calculate Confidence
        confidence = (TP / len(subgroup_data)) if len(subgroup_data) > 0 else 0

        # Calculate WRAcc
        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))
        
        support = TP / len(data)

        quality_measures.append(subgroup_mean - overall_mean)
        coverage_list.append(coverage)
        support_list.append(support)
        wracc_list.append(wracc)
        significance_list.append(p_value)
        confidence_list.append(confidence)

    frequent_itemsets['quality'] = quality_measures
    frequent_itemsets['coverage'] = coverage_list
    frequent_itemsets['support'] = support_list
    frequent_itemsets['WRAcc'] = wracc_list
    frequent_itemsets['Significance'] = significance_list
    frequent_itemsets['Confidence'] = confidence_list

    # Sort subgroups based on quality
    sorted_subgroups = frequent_itemsets.sort_values(by='quality', ascending=False)

    # Prune redundant subgroups to get a set of non-redundant subgroups
    non_redundant_subgroups = []
    for _, row in sorted_subgroups.iterrows():
        is_redundant = False
        for nr_subgroup in non_redundant_subgroups:
            if row['itemsets'].issubset(nr_subgroup):
                is_redundant = True
                break
        if not is_redundant:
            non_redundant_subgroups.append(row['itemsets'])

    non_redundant_subgroups_df = sorted_subgroups[sorted_subgroups['itemsets'].isin(non_redundant_subgroups)]

    result_metrics = {
        'Average Quality': np.mean(non_redundant_subgroups_df.quality),
        'Average Coverage': np.mean(non_redundant_subgroups_df.coverage),
        'Average Support': np.mean(non_redundant_subgroups_df.support),
        'Average WRAcc': np.mean(non_redundant_subgroups_df.WRAcc),
        'Average Significance': np.mean(non_redundant_subgroups_df.Significance),
        'Average Confidence': np.mean(non_redundant_subgroups_df.Confidence),
        'Number of Subgroups': len(non_redundant_subgroups_df["quality"]),
        'Average Length of Subgroups': np.mean([len(rule) for rule in frequent_itemsets["itemsets"]]),
    }

    return result_metrics


def nmeef_sd(data, target_column, n_generations=10, population_size=100):
    """
    Final NMEEF-SD evolutionary algorithm for subgroup discovery with crossover and mutate functions.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.
    - n_generations: int, number of generations.
    - population_size: int, size of the rule population.

    Returns:
    - pd.DataFrame containing rules (as interpretable conditions) and their quality measures.
    """

    # Helper functions for the evolutionary processes
    def crossover(parent1, parent2):
        """One-point crossover."""
        point = random.randint(0, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def mutate(rule):
        """Bit-flip mutation."""
        index = random.randint(0, len(rule) - 1)
        new_bit = '1' if rule[index] == '0' else '0'
        return rule[:index] + new_bit + rule[index + 1:]

    def initialize_population():
        """Initialize a diverse random population of binary-coded rules."""
        return ["".join(random.choice(['0', '1']) if random.random() < 0.5 else '0' for _ in range(data.shape[1] - 1))
                for _ in range(population_size)]

    def evaluate_rule(rule):
        """Evaluate the quality of a rule based on *metrics*."""
        subgroup_data = data[np.logical_and.reduce(
            [(data[columns[idx]] == 1) if bit == '1' else (data[columns[idx]] == 0) for idx, bit in enumerate(rule)])]
        if len(subgroup_data) == 0:
            return 0
        subgroup_mean = subgroup_data[target_column].mean()

        non_subgroup_data = data[~data.index.isin(subgroup_data.index)]
        _, significance = ranksums(subgroup_data[target_column], non_subgroup_data[target_column])

        TP = len(subgroup_data[subgroup_data[target_column] == 1])
        FP = len(subgroup_data[subgroup_data[target_column] == 0])

        # Calculate Confidence
        confidence = (TP / len(subgroup_data)) if len(subgroup_data) > 0 else 0

        # Calculate WRAcc
        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))
        coverage = len(subgroup_data) / len(data)
        support = TP / len(data)
        quality = (subgroup_mean - overall_mean)
        return (quality, coverage, support, wracc, significance, confidence, rule)

    def binary_to_conditions(binary_rule, columns):
        """Convert binary rule to interpretable conditions."""
        conditions = [col for idx, col in enumerate(columns) if binary_rule[idx] == '1']
        return ', '.join(conditions)

    columns = list(data.drop(columns=[target_column]).columns)
    overall_mean = data[target_column].mean()

    # Initialize population and storage for best rules and their quality
    population = initialize_population()
    best_rules = []
    best_metrics = pd.DataFrame(columns = ["quality", "coverage", "support", "WRAcc", "significance", "confidence"])

    # Evolutionary process
    for _ in range(n_generations):
        # Evaluate population
        rule_metrics = []
        for rule in population:
            temp = evaluate_rule(rule)
            if type(temp) == tuple:
                rule_metrics.append(temp)
        rule_metrics = pd.DataFrame(rule_metrics, columns = ["quality", "coverage", "support", "WRAcc", "significance", "confidence", "rule"])

        # Store best rules and their qualities
        sorted_population = list(rule_metrics.sort_values("quality")["rule"])
        best_rules.extend(sorted_population[:5])
        best_metrics = pd.concat([best_metrics, rule_metrics.sort_values("quality").head(5)[["quality", "coverage", "support", "WRAcc", "significance", "confidence"]]], ignore_index=True)


        # Select parents and produce offspring
        parents = sorted_population[:population_size // 2]
        # print(parents)
        offspring = []
        for i in range(0, len(parents) // 2 - 2, 2):
            # print(i)
            offspring1, offspring2 = crossover(parents[i], parents[i+1])
            offspring.append(mutate(offspring1))
            offspring.append(mutate(offspring2))

        # Form new population
        population = parents + offspring

    # Convert best rules into DataFrame with interpretable conditions
    # rules_as_conditions = [binary_to_conditions(rule, columns) for rule in best_rules]

    result_metrics = {
        'Average Quality': np.mean(best_metrics.quality),
        'Average Coverage': np.mean(best_metrics.coverage),
        'Average Support': np.mean(best_metrics.support),
        'WRAcc': np.mean(best_metrics.WRAcc),
        'Significance': np.mean(best_metrics.significance),
        'Confidence': np.mean(best_metrics.confidence),
        'Number of Subgroups': len(best_rules),
        'Average Length of Subgroups': np.mean([str(rule).count("1") for rule in best_rules]),
    }

    return result_metrics


def apriori_sd(data, target_column, min_support=0.1, metric="lift", min_threshold=1):
    """
    Subgroup Discovery (SD) using the APRIORI algorithm with binarized data.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.
    - min_support: float, minimum support for the frequent itemsets.
    - metric: str, metric to evaluate if a rule is of interest.
    - min_threshold: float, minimum threshold for the metric.

    Returns:
    - pd.DataFrame containing rules (as interpretable conditions) and their quality measures.
    """

    def calculate_wracc(antecedent, consequent, data):
        # Calculate TP, FP, and N (total instances)
        # print(rule)
        
        # antecedent = data[antecedent]
        # consequent = data[consequent]
        # TP = (antecedent & consequent).sum()
        # FP = (antecedent & ~consequent).sum()
        N = len(data)

        # antecedent = list(antecedent)[0]

        subgroup_data = data[antecedent]
        # print(antecedent)
        # print("hi")
        # print(subgroup_data)

        consequent = list(consequent)[0]

        TP = len(subgroup_data[subgroup_data[consequent] == 1])
        FP = len(subgroup_data[subgroup_data[consequent] == 0])
        
        # Calculate WRAcc
        WRAcc = TP / N - (TP + FP / N) * (len(data[data[consequent] == 1] / len(data)))
        
        return WRAcc
    
    def calculate_significance(antecedent, consequent, data):
    # Extract the subgroup's conditions

        rule = antecedent & consequent
        subgroup_data = data[data[rule]]
        non_subgroup_data = data - subgroup_data

        _, p_value = ranksums(subgroup_data[consequent], non_subgroup_data[consequent])

        return p_value

    # Convert numeric columns to binary (0/1) based on a threshold
    # Here, I'm using the median as the threshold, but you can adjust it
    for column in data.select_dtypes(['int64', 'float64']).columns:
        threshold = data[column].median()
        data[column] = (data[column] > threshold).astype(int)

    # Convert data into one-hot encoded format suitable for apriori
    data_encoded = pd.get_dummies(data)

    # Find frequent itemsets using the apriori algorithm
    frequent_itemsets = apriori(data_encoded, min_support=min_support, use_colnames=True)

    # Generate association rules from the frequent itemsets
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    # Filter rules where the consequent is the target column
    target_rules = rules[rules['consequents'] == frozenset({target_column})]
    
    WRAcc_list = []
    significance_list = []
    data = data.dropna()
    for i in range(len(target_rules)):
        # print(type(target_rules["antecedents"]))
        # wracc = calculate_wracc(list(target_rules['antecedents'])[i], list(target_rules['consequents'])[i], data)
        significance = calculate_significance(list(target_rules['antecedents'])[i], list(target_rules['consequents'])[i], data)
        # print(wracc)
        # WRAcc_list.append(wracc)
        significance_list.append(significance)

    # Rank rules based on the lift metric (or any other chosen metric)
    target_rules = target_rules.sort_values(by=metric, ascending=False)

    # Convert antecedents and consequents from frozenset to string for readability
    target_rules['rule'] = target_rules['antecedents'].apply(lambda x: ' AND '.join(list(x)))
    target_rules['consequent'] = target_rules['consequents'].apply(lambda x: list(x)[0])
    target_rules['coverage'] = target_rules['support'] / target_rules['antecedent support']

    # print(target_rules['rule'])

    result_metrics = {
        'Average Quality': None,
        'Average Coverage': np.mean(target_rules.coverage),
        'Average Support': np.mean(target_rules.support),
        'WRAcc': None,
        'Significance': None,
        'Confidence': None,
        # 'WRAcc': np.mean(WRAcc_list),
        # 'Significance': np.mean(significance_list),
        # 'Confidence': np.mean(target_rules.confidence),
        'Number of Subgroups': len(target_rules),
        'Average Length of Subgroups': np.mean([len(rule) for rule in target_rules["antecedents"]]),
    }

    # Return a dataframe with the rules and their quality measures
    return result_metrics


