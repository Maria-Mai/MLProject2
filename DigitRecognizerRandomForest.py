"""
this scripts uses a simple decision tree classifier
and a random forest classifier
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

def custom_decision_tree(X_train, y_train, seed, scoring):
    """
    decision tree classifier evaluated with cross vlaidation
    :param X_train: x training data
    :param y_train: y training data
    :param seed: seed
    :param scoring: measures that will be taken
    :return: score (measures)
    """
    clf_tree = DecisionTreeClassifier(random_state=seed)
    score = cross_validate(clf_tree, X_train, y_train, scoring=scoring, cv=3, return_train_score=True)
    return score

def custom_random_forest(X_train, y_train, criterion, max_depth, scoring, seed):
    """
    random forest classifier evaluated with cross validation
    :param X_train: x train data
    :param y_train: y train data
    :param criterion: entropy or gini
    :param max_depth: max depth of the tree
    :param scoring: measures that will be taken
    :param seed: seen
    :return: score (measures)
    """
    clf_forest = RandomForestClassifier(n_estimators=10, criterion= criterion, max_depth=max_depth, random_state=seed)
    score = cross_validate(clf_forest, X_train, y_train, scoring=scoring, cv=3, return_train_score=True)
    return score