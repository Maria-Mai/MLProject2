"""
This script runs 3 different classifiers with
different hyperparameters and selects the best one.
Then it trains the best classifier again
and gets the measures for unseen test data.
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import DigitRecognizerSVM as d_svm
import DigitRecognizerRandomForest as d_forest
from keras.utils import to_categorical
import DigitRecognizerCNN as d_cnn


def calculate_mean(current_score, key):
    """
    Calculates the mean for cross validated measures
    :param current_score: score of the current classifier
    :param key: which measure
    :return: mean
    """
    mean = current_score[key]
    return (sum(mean) / len(mean))

def append_pdf(pdf, name, a_list, plot_figure, x_values=[""]):
    """
    Writes measures of the classifiers to a pdf file
    :param pdf: pdf file
    :param name: name of classifier
    :param a_list: list with scores
    :param plot_figure: if a graph should be printed
    :param x_values: x values for the graph
    """

    acc_val_list = list()
    acc_train_list = list()
    f1_val_list = list()
    f1_train_list = list()

    for i in range(len(a_list)):
        train_acc = calculate_mean(a_list[i], "train_accuracy")
        acc_train_list.append(train_acc)

        val_acc = calculate_mean(a_list[i], "test_accuracy")
        acc_val_list.append(val_acc)

        train_f1 = calculate_mean(a_list[i], "train_f1_macro")
        f1_train_list.append(train_f1)

        val_f1 = calculate_mean(a_list[i], "test_f1_macro")
        f1_val_list.append(val_f1)

        fit_time = calculate_mean(a_list[i], "fit_time")
        score_time = calculate_mean(a_list[i], "score_time")

        a_page = plt.figure(figsize=(11.69, 8.27))
        a_page.clf()

        txt = "Classifier: " + name + "kp: " + str(x_values[i]) + "\n"\
              "train accuracy: " + str(train_acc) + "    " + "val accuracy: " + str(val_acc) + "\n" + \
              "train f1: " + str(train_f1) + "    " + "val f1: " + str(val_f1)  + "\n" + \
            "fit_time: " + str(fit_time) + "    " + "score time: " + str(score_time)

        a_page.text(0.1, 0.1, txt, size=20)
        pdf.savefig()
        plt.close()

    if(plot_figure):
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.plot(x_values, acc_val_list, label="acc_val")
        plt.plot(x_values, acc_train_list, label="acc_train")
        plt.plot(x_values, f1_val_list, label="f1_val")
        plt.plot(x_values, f1_train_list, label="f1_train")
        plt.legend(loc='best')
        plt.grid(True)
        plt.title(name)
        pdf.savefig()

#-----------------------------------------------------------
pdf = PdfPages("report2.pdf")

#make data reproducible
seed = 0 # todo change to 42

#load data
print("Start read in data")

X = pd.read_csv("handwritten_digits_images.csv", header=None)
print(X.shape)
X = X.values.reshape(X.shape[0],28,28)
print(X.shape)

y = pd.read_csv("handwritten_digits_labels.csv", header=None)
print(y.shape)
print(y[0].value_counts())

plt.imshow(X[0], cmap="Greys")
#plt.show()
print("End read in data")

#normalize the images.
X = (X / 255) - 0.5

print("put aside test data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

X_train_2d = X_train.reshape(X_train.shape[0], X_train.shape[1] ** 2)
y_train_1d = y_train.values.flatten()

#scores that will be measured
scoring = ['accuracy', 'f1_macro']
list_f1_scores = list()
list_acc_scores = list()


#use of classifiers

#svm
print("svm")

#takes very long, if u still want to run it uncomment it
"""
#svm without pca
svm_list = list()
score = d_svm.custom_SVM(X_train_2d, y_train_1d, penalty=0.5, seed=0, scoring=scoring)
list_f1_scores.append(("decision tree", calculate_mean(score, "test_f1_macro")))
list_acc_scores.append(("decision tree", calculate_mean(score, "test_accuracy")))
svm_list.append(score)
append_pdf(pdf, "svm without pca C: 0.5", svm_list, False)"""

#x values for the graph
x_values = [0.3,0.7,1, 1.5,2]


svm_hyperparameter_c5 = [{"n_components": 5, "C": 0.3},
                         {"n_components": 5, "C": 0.7},
                         {"n_components": 5, "C": 1},
                         {"n_components": 5, "C": 1.5},
                         {"n_components": 5, "C": 2}]
svm_c1_list = list()
for hyperpara in svm_hyperparameter_c5:
    print(hyperpara)
    score = d_svm.improved_SVM(X_train_2d, y_train_1d, hyperpara["C"], hyperpara["n_components"], seed, scoring)
    list_f1_scores.append((hyperpara, calculate_mean(score, "test_f1_macro")))
    list_acc_scores.append((hyperpara, calculate_mean(score, "test_accuracy")))
    svm_c1_list.append(score)
append_pdf(pdf, "svm with pca n_components=5", svm_c1_list, True, x_values)


svm_hyperparameter_c15 = [{"n_components": 15, "C": 0.3},
                        {"n_components": 15, "C": 0.7},
                          {"n_components": 15, "C": 1},
                          {"n_components": 15, "C": 1.5},
                          {"n_components": 15, "C": 2}]

svm_c3_list = list()
for hyperpara in svm_hyperparameter_c15:
    print(hyperpara)
    score = d_svm.improved_SVM(X_train_2d, y_train_1d, hyperpara["C"], hyperpara["n_components"], seed, scoring)
    list_f1_scores.append((hyperpara, calculate_mean(score, "test_f1_macro")))
    list_acc_scores.append((hyperpara, calculate_mean(score, "test_accuracy")))
    svm_c3_list.append(score)
append_pdf(pdf, "svm with pca n_components=0.3", svm_c3_list, True, x_values)


#---------------------------------------------------------------------

#forest
print("decision tree")

tree_list = list()
score = d_forest.custom_decision_tree(X_train_2d, y_train_1d, seed, scoring)
list_f1_scores.append(("decision tree", calculate_mean(score, "test_f1_macro")))
list_acc_scores.append(("decision tree", calculate_mean(score, "test_accuracy")))
tree_list.append(score)
append_pdf(pdf, "simple decision tree", tree_list, False)

x_values = [10,20,30,40,50]
forest_gini_dicts = [{"criterion": "gini", "max_depth": 10},
                     {"criterion": "gini", "max_depth": 20},
                     {"criterion": "gini", "max_depth": 30},
                     {"criterion": "gini", "max_depth": 40},
                     {"criterion": "gini", "max_depth": 50}]

forest_gini_list = list()
for hyperpara in forest_gini_dicts:
    print(hyperpara)
    score = d_forest.custom_random_forest(X_train_2d, y_train_1d, hyperpara["criterion"], hyperpara["max_depth"], scoring, seed)
    list_f1_scores.append((hyperpara, calculate_mean(score, "test_f1_macro")))
    list_acc_scores.append((hyperpara, calculate_mean(score, "test_accuracy")))
    forest_gini_list.append(score)
append_pdf(pdf, "random forest gini", forest_gini_list, True,x_values)

forest_entopy_list = list()
forest_entropy_dicts = [{"criterion": "entropy", "max_depth": 10},
                       {"criterion": "entropy", "max_depth": 20},
                       {"criterion": "entropy", "max_depth": 30},
                       {"criterion": "entropy", "max_depth": 40},
                       {"criterion": "entropy", "max_depth": 50}]
for hyperpara in forest_entropy_dicts:
    print(hyperpara)
    score = d_forest.custom_random_forest(X_train_2d, y_train_1d, hyperpara["criterion"], hyperpara["max_depth"], scoring, seed)
    list_f1_scores.append((hyperpara, calculate_mean(score, "test_f1_macro")))
    list_acc_scores.append((hyperpara, calculate_mean(score, "test_accuracy")))
    forest_entopy_list.append(score)
append_pdf(pdf, "random forest entropy", forest_entopy_list, True, x_values)


#--------------------------------------------------------

#cnn

X_train_conv = np.expand_dims(X_train, 4)
y_train_conv = to_categorical(y_train)


x_values = [0.001, 0.01]

cnn_hyperparameter_pool = [{"pooling": True, "learning_rate":0.001},
                            {"pooling": True, "learning_rate":0.01}]
cnn_pool_list = list()
for hyperpara in cnn_hyperparameter_pool:
    print(hyperpara)
    score = d_cnn.custom_CNN(X_train_conv, y_train_conv, hyperpara["pooling"], hyperpara["learning_rate"])
    list_acc_scores.append((hyperpara, score[0][1]))
    list_f1_scores.append((hyperpara, score[0][2]))
    score = {"train_accuracy": [score[1]], "test_accuracy": [score[0][1]], "train_f1_macro": [score[2]],
             "test_f1_macro": [score[0][2]], "fit_time": [0], "score_time": [0]}
    cnn_pool_list.append(score)
append_pdf(pdf, "cnn with pooling", cnn_pool_list, True, x_values)

cnn_hyperparameter_non_pool= [{"pooling": False, "learning_rate":0.001},
                            {"pooling": False, "learning_rate":0.01}]
cnn_non_pool_list = list()
for hyperpara in cnn_hyperparameter_non_pool:
    print(hyperpara)
    score = d_cnn.custom_CNN(X_train_conv, y_train_conv, hyperpara["pooling"], hyperpara["learning_rate"])
    print(score)
    list_acc_scores.append((hyperpara, score[0][1]))
    list_f1_scores.append((hyperpara, score[0][2]))
    score = {"train_accuracy": [score[1]], "test_accuracy": [score[0][1]], "train_f1_macro": [score[2]],
             "test_f1_macro": [score[0][2]], "fit_time": [0], "score_time": [0]}
    cnn_non_pool_list.append(score)
append_pdf(pdf, "cnn with pooling", cnn_pool_list, True, x_values)

#-------------------------------------------------------

#select best classifier by f1-score
best_classifier_f1 = max(list_f1_scores, key=lambda item:item[1])
print(best_classifier_f1)

#select best by accuracy?
best_classifier_acc = max(list_acc_scores, key=lambda item:item[1])
print(best_classifier_acc)

#test data metrics



pdf.close()
