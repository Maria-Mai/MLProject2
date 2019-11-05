"""
this script implements a convolutional neuronal network
"""
#convolutional network
import numpy as np
np.random.seed(0)
from keras import Sequential
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def custom_CNN(X_train, y_train, pooling, learning_rate):
    """

    :param X_train:
    :param y_train:
    :param pooling:
    :param learning_rate:
    :return:
    """
    early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=1)
    clf_cnn = Sequential()
    clf_cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    if (pooling):
        clf_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    clf_cnn.add(Conv2D(32, kernel_size=3, activation='relu'))
    clf_cnn.add(Flatten())
    clf_cnn.add(Dense(10, activation='softmax'))

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=0)
    return fit_and_evaluate(clf_cnn, X_train, y_train, X_val, y_val, learning_rate, early_stopping)

def fit_and_evaluate(model, X_train, y_train, X_val, y_val, learning_rate, early_stopping):
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", f1_m])
    hist = model.fit(X_train, y_train, epochs=2, batch_size=256, callbacks=[early_stopping], shuffle=False) #todo epoch
    return model.evaluate(X_val, y_val), hist.history.get("accuracy")[-1], hist.history.get("f1_m")[-1],

def f1_m(y_true, y_pred):
    """
    calculates the f1 score for the cnn
    :param y_true: y given data
    :param y_pred: y predicted data
    :return: f1 score
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def precision_m(y_true, y_pred):
    """
    calculates the precison for the f1 score
    :param y_true: y given data
    :param y_pred: y predicted data
    :return: precison
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_m(y_true, y_pred):
    """
    calulates the recall for the f1 score
    :param y_true: y given data
    :param y_pred: y predicted data
    :return: recall
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall