import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras.optimizers import Adam, SGD
from sklearn.utils.class_weight import compute_class_weight

def balanced_recall(y_true, y_pred):
    """
    Computes the average per-column recall metric
    for a multi-class classification problem
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
    recall = true_positives / (possible_positives + K.epsilon())
    balanced_recall = K.mean(recall)
    return balanced_recall

def Model(feature_num):
    # Create model
    ann = Sequential()
    neurons_1 = 500  # int(feature_num * 2 - (feature_num * 2) % 100) # let it be a multiple of 100
    neurons_2 = 500
    neurons_3 = 500

    ann.add(Dense(neurons_1, input_dim=feature_num, activation='relu'))  # leaky_relu
    ann.add(Dropout(0.5))
    ann.add(BatchNormalization())
    ann.add(Dense(neurons_2, activation='relu'))
    ann.add(Dropout(0.5))
    ann.add(BatchNormalization())
    ann.add(Dense(neurons_3, activation='relu'))
    ann.add(Dropout(0.5))
    ann.add(Dense(4, activation='softmax'))

    # learning rate decay
    initial_lr = 0.01
    adam = Adam(lr = initial_lr)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # Compile model
    ann.compile(optimizer= 'adam',  loss='categorical_crossentropy', metrics=[balanced_recall])
    return ann

def neurNet_classifier(train_x, train_y, test_x):
    # Feature Scaling, 0 mean and 1 standard variance
    scalar = StandardScaler()
    train_x = scalar.fit_transform(train_x.astype('float64'))
    test_x = scalar.transform(test_x.astype('float64'))

    # convert the y label to categorical vector size of n,
    train_y_vector = np_utils.to_categorical(train_y)

    # set class weights according to the imbalance of the data
    weights = compute_class_weight('balanced', np.array([0, 1, 2, 3]), np.asarray(train_y.squeeze()))
    class_weighting = {0: weights[0], 1: weights[1], 2: weights[2], 3:weights[3]}
    print(class_weighting)

    # learning rate decay

    ann1 = Model(train_x.shape[1])
    ann1.fit(train_x, train_y_vector, epochs=100, batch_size=128, validation_split=0.1, class_weight=class_weighting,
             verbose=True)

    # Predict unlabeled
    y_predict_test = np.argmax(ann1.predict(test_x), axis=1)
    return y_predict_test