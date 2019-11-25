import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, BatchNormalization, Lambda, Layer, GaussianNoise, Conv1D, MaxPooling1D, Flatten
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score
from keras.utils import np_utils
from sklearn.svm import NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture  import BayesianGaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier


def CNNModel(hb_length = 180,cutoff = 5000):
    # Create model
    act = 'sigmoid'
    d_loss = 0.45

    input_hb_avg = Input(shape = (hb_length,))
    input_hb_var = Input(shape = (hb_length,))
    input_tf     = Input(shape = (cutoff, ))

    x = Conv1D(32, kernel_size = 50, activation = act)(input_hb_avg)
    x = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (x)
    x = Conv1D(64, kernel_size= 20, activation=act)(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = Model(inputs = input_hb_avg, outputs = x )

    y = Conv1D(32, kernel_size = 50, activation = act)(input_hb_var)
    y = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (y)
    y = Conv1D(64, kernel_size= 20, activation=act)(y)
    y = MaxPooling1D(pool_size=3, strides=2)(y)
    y = Model(inputs = input_hb_var, outputs = y )

    combined = concatenate( x.output, y.output)

    z = Conv1D(128, kernel_size= 50, activation=act)(combined)
    z = MaxPooling1D(pool_size=3, strides=2)(z)
    z = Model(inputs = [x.output, y.output], outputs = z)

    q = Conv1D(32, kernel_size = 100, activation = act)(input_tf)
    q = MaxPooling1D(pool_size=3, strides=2)(q)
    q = Conv1D(64, kernel_size= 50, activation=act)(q)
    q = MaxPooling1D(pool_size=3, strides=2)(q)
    q = Model(inputs = input_tf, outputs = q )

    combined_2 = concatenate( z.output, q.output)

    r = Dense(800, activation = act) (combined_2)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)
    r = Dense(400, activation = act) (r)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)
    r = Dense(2, activation = 'softmax')(r)
    
    model = Model(inputs = [x.input, y.input, q.input], outputs = r)

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['categorical_accuracy'])    

    return model

def label(y_pred) :
    y_labeled = []
    for i in y_pred:
        y_labeled.append(int(np.where(i == np.amax(i))[0]))
    return y_labeled   

def CNN_predict(test_x_hb, test_x_tf, y_train test_x_hb, test_x_tf) :


def rfClassify(train_x, train_y, test_x, test_y, class_weights, estimators = 10, predict = False) :
    rf = RandomForestClassifier(max_depth = 10, n_estimators = estimators, class_weight = class_weights)
    rf.fit(train_x, train_y.reshape(-1,))
    y_pred = rf.predict(test_x)
    if ( predict == False) :
        score =  balanced_accuracy_score(test_y.reshape(-1,1), y_pred)
        print('Random Forest produced a score of:', score, ' for ', estimators, ' estimators.')
        return score
    else :
        return y_pred

def bgmClassify(train_x, train_y, test_x, test_y, predict = False) :
    rf = GaussianProcessClassifier(max_iter_predict = 10000, n_restarts_optimizer = 10, n_jobs = -1)
    rf.fit(train_x, train_y.reshape(-1,))
    y_pred = rf.predict(test_x)
    if ( predict == False) :
        score =  balanced_accuracy_score(test_y.reshape(-1,1), y_pred)
        print('GaussianProcess produced a score of:', score)
        return score
    else :
        return y_pred



def svmClassify(train_x, train_y, test_x, test_y, class_weights, kernel = 'rbf', gamma = 'scale', nu = 0.04, predict = False):

    clf = NuSVC(nu= nu,gamma= gamma,decision_function_shape = 'ovr',
        probability = True, kernel=kernel, class_weight = class_weights)

    clf.fit(train_x, train_y.reshape(-1,))
    y_pred = clf.predict_proba(test_x)
    if ( predict == False) :
        score =  balanced_accuracy_score(test_y.reshape(-1,1), label(y_pred))
        print('svm produced a score of:', score, ' for ', len(train_x[0]), ' variables.')
        return score
    else :
        return y_pred

def svcClassify(train_x, train_y, test_x, test_y, class_weights,
    kernel = 'rbf', gamma = 'scale', _C = 1.0, predict = False) :
    
    clf = SVC(C = _C,gamma= gamma, kernel=kernel, class_weight = class_weights)
    clf.fit(train_x, train_y.reshape(-1,))
    y_pred = clf.predict(test_x)
    if ( predict == False) :
        score =  balanced_accuracy_score(test_y.reshape(-1,1), y_pred)
        print('svc produced a score of:', score, ' for a gamma of', gamma, ' and a C of ', _C)
        return score
    else :
        return y_pred




def simpleClassify(train_x, train_y, test_x, test_y, sample_weights,  neurons = 1024, predict = False) :

    # train labeled

    ann1 = Model(train_x[0].size, neurons)
    epochs = 3000
    bestepoch = 700
    
    naive = ann1.get_weights()
    bestweights = ann1.get_weights()
    
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.08, stratify = train_y)
    y_train_oneh = np_utils.to_categorical(y_train)

    oldscore = 0.1
    bestepoch = 0
    UPDATE = 0
    for i in range(0,epochs) :
        ann1.fit(X_train, y_train_oneh, verbose = 1, epochs=1, batch_size=64, class_weight = sample_weights)
        y_pred = ann1.predict(X_test)
        score = balanced_accuracy_score(y_test, label(y_pred))
        UPDATE += 1
        if score > oldscore :
            oldscore = score
            print('new best score: ', score, ' at epoch', i)
            bestepoch = i
            bestweights = ann1.get_weights()
            UPDATE = 0
        if score > 0.75 :
            break

    ann1.set_weights(bestweights)
    ann1.fit(X_test, np_utils.to_categorical(y_test), verbose = 1, epochs = 30, batch_size = 128,  class_weight = sample_weights)
    y_pred = ann1.predict(test_x)
#    print(y_pred)
    if predict == False :
        score = balanced_accuracy_score(test_y, label(y_pred))
        print('model evaluated, had a score of ' , score)
        return np.max([0.25,score])
    else:
        return y_pred
