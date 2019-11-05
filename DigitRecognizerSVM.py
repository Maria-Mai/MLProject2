"""
This script implements the usage of the svm
from the sklearn package together with a pca
"""

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate


def custom_SVM(X_train, y_train, penalty, seed, scoring):
    """
    svm classifier with 3 fold cross validation
    :param X_train: x training data
    :param y_train: y training data
    :param penalty: penalty term
    :param seed: seed
    :param scoring: scoring
    :return: score (measure)
    """
    clf_svm = svm.SVC(C=penalty, kernel="rbf", gamma='auto', random_state=seed)
    score = cross_validate(clf_svm, X_train,y_train, scoring=scoring, cv=3, return_train_score=True)
    return score

def improved_SVM(X_train, y_train, penalty, n_components, seed, scoring):
    """
    svm using pca before the svm step
    :param X_train: x training data
    :param y_train: y training data
    :param penalty: penalty term C
    :param n_components: components of the svm
    :param seed: seed
    :param scoring: scoring
    :return: score (measures)
    """
    X_train = custom_PCA(n_components, X_train, seed)
    score = custom_SVM(X_train, y_train, penalty, seed, scoring)
    return score

def custom_PCA(n_components, X_train, seed):
    """
    pca before doing using the svm classifier
    :param n_components: number of components
    :param X_train: x data
    :param seed: seed
    :return: transformed data
    """
    a_pca = PCA(n_components=n_components, random_state=seed).fit(X_train)
    X_train = a_pca.transform(X_train)
    return X_train