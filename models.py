import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import Orange
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

    # Convert rules into DataFrame
    rules_df = pd.DataFrame({
        'rule': rules,
        'quality': qualities
    })

    return rules_df.drop_duplicates().sort_values(by='quality', ascending=False)


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
        rules.append((rule_str, rule.quality))

    # Convert rules into DataFrame
    rules_df = pd.DataFrame(rules, columns=['rule', 'quality'])

    return rules_df.sort_values(by='quality', ascending=False)


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

    for _, row in frequent_itemsets.iterrows():
        subgroup_data = data[np.logical_and.reduce([data[col] for col in row['itemsets']])]
        subgroup_mean = subgroup_data[target_column].mean()
        quality_measures.append(subgroup_mean - overall_mean)

    frequent_itemsets['quality'] = quality_measures

    # Rank subgroups based on quality
    ranked_subgroups = frequent_itemsets.sort_values(by='quality', ascending=False)

    return ranked_subgroups


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
    import numpy as np

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

    for _, row in frequent_itemsets.iterrows():
        subgroup_data = data[np.logical_and.reduce([data[col] for col in row['itemsets']])]
        subgroup_mean = subgroup_data[target_column].mean()
        quality_measures.append(subgroup_mean - overall_mean)

    frequent_itemsets['quality'] = quality_measures

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

    return non_redundant_subgroups_df


def nmeef_sd(data, target_column, n_generations=50, population_size=100):
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

    # Helper functions for the evolutionary process
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
        """Evaluate the quality of a rule based on difference in means and subgroup size."""
        subgroup_data = data[np.logical_and.reduce(
            [(data[columns[idx]] == 1) if bit == '1' else (data[columns[idx]] == 0) for idx, bit in enumerate(rule)])]
        if len(subgroup_data) == 0:
            return 0
        subgroup_mean = subgroup_data[target_column].mean()
        return (subgroup_mean - overall_mean) ** 2 * len(subgroup_data) / len(data)

    def binary_to_conditions(binary_rule, columns):
        """Convert binary rule to interpretable conditions."""
        conditions = [col for idx, col in enumerate(columns) if binary_rule[idx] == '1']
        return ', '.join(conditions)

    columns = list(data.drop(columns=[target_column]).columns)
    overall_mean = data[target_column].mean()

    # Initialize population and storage for best rules and their quality
    population = initialize_population()
    best_rules = []
    best_qualities = []

    # Evolutionary process
    for _ in range(n_generations):
        # Evaluate population
        fitness_values = [evaluate_rule(rule) for rule in population]

        # Store best rules and their qualities
        sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
        best_rules.extend(sorted_population[:5])
        best_qualities.extend(sorted(zip(fitness_values, population), reverse=True)[:5])

        # Select parents and produce offspring
        parents = sorted_population[:population_size // 2]
        offspring = []
        for i in range(0, population_size // 2, 2):
            offspring1, offspring2 = crossover(parents[i], parents[i + 1])
            offspring.append(mutate(offspring1))
            offspring.append(mutate(offspring2))

        # Form new population
        population = parents + offspring

    # Convert best rules into DataFrame with interpretable conditions
    rules_as_conditions = [binary_to_conditions(rule, columns) for rule in best_rules]
    best_rules_df = pd.DataFrame({
        'rule': rules_as_conditions,
        'quality': [quality for quality, _ in best_qualities]
    })

    return best_rules_df.sort_values(by='quality', ascending=False).drop_duplicates()


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

    # Rank rules based on the lift metric (or any other chosen metric)
    target_rules = target_rules.sort_values(by=metric, ascending=False)

    # Convert antecedents and consequents from frozenset to string for readability
    target_rules['rule'] = target_rules['antecedents'].apply(lambda x: ' AND '.join(list(x)))
    target_rules['consequent'] = target_rules['consequents'].apply(lambda x: list(x)[0])

    # Return a dataframe with the rules and their quality measures
    return target_rules[['rule', 'consequent', 'support', 'confidence', 'lift']]


