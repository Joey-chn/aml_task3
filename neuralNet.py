import numpy as np
import pandas as pd
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, BatchNormalization, Lambda, Layer, GaussianNoise, Conv1D, MaxPooling1D, Flatten, Input, Concatenate, Reshape
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from keras.utils import np_utils
from sklearn.utils import class_weight
import gc
from random import sample
from sklearn.preprocessing import StandardScaler

#optimal loss function, but not exactly the right one...

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def get_class_weights(y_train) :
    y_integers = np.argmax(np_utils.to_categorical(y_train), axis=1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    return dict(enumerate(class_weights))
    #return class_weights

def CNNModel3(classes, hb_length = 180,cutoff = 5000, hrv_len = 20):
    # Create model
    act = 'sigmoid'
    act2 = 'sigmoid'
    d_loss = 0.55

    input_hb_avg = Input(shape = (hb_length,1))
    input_hb_var = Input(shape = (hb_length,1))
    input_tf     = Input(shape = (cutoff,1))
    input_hrv =    Input(shape = (hrv_len,))

    x0 = GaussianNoise(30.0)(input_hb_avg)
    x = Conv1D(16, kernel_size = 50, activation = act)(x0)
    x1 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (x)
    x2 = Conv1D(32, kernel_size= 25, activation=act)(x1)
    x3 = MaxPooling1D(pool_size=3, strides=2)(x2)
    x4 = Model(inputs = input_hb_avg, outputs = x3 )

    y0 = GaussianNoise(30.0)(input_hb_var)
    y = Conv1D(16, kernel_size = 50, activation = act)(y0)
    y1 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (y)
    y2 = Conv1D(32, kernel_size= 25, activation=act)(y1)
    y3 = MaxPooling1D(pool_size=3, strides=2)(y2)
    y4 = Model(inputs = input_hb_var, outputs = y3 )

    combined = Concatenate()( [x4.output, y4.output])

    z = Conv1D(32, kernel_size= 20, activation=act)(combined)
    z1 = MaxPooling1D(pool_size=3, strides=2)(z)
    z2 = Model(inputs = [input_hb_avg, input_hb_var], outputs = z1)

    q0 = GaussianNoise(200.0)(input_tf)
    q = Conv1D(16, kernel_size = 200, activation = act)(q0)
    q1 = MaxPooling1D(pool_size=8, strides=6)(q)
    q2 = Conv1D(32, kernel_size= 150, activation=act)(q1)
    q3 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (q2)
    q30 = Conv1D(32, kernel_size= 100, activation=act)(q3)
    q31 = MaxPooling1D(pool_size=4, strides=3)(q30)
    q4 = Flatten()(q31)
    q5 = Dense(512, activation = act2)(q4)

    q5 = BatchNormalization()(q5)
    q6 = Reshape((16,32))(q5)
    q4 = Model(inputs = input_tf, outputs = q6 )

    bypass = Flatten()(q2)
    bypass = Dense(256, activation = act2)(bypass)
    bypass = Dropout(d_loss)(bypass)
    bypass = BatchNormalization()(bypass)


    combined_2 = Concatenate()([ z2.output, q4.output])

    r0 = Flatten() (combined_2)

    r = Dense(1024, activation = act2) (r0)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)

    r = Dense(512, activation = act2) (r)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)

    s = Dense(512, activation = act2)(input_hrv)
    s = Dropout(d_loss)(s)
    s = BatchNormalization()(s)


    combined_3= Concatenate()([r, s])
    
    r = Dense(256, activation = act2) (combined_3)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)
    r = Add()([r, bypass])
    
    r3 = Dense(128, activation = act2) (r)
    r4 = Dropout(d_loss)(r3)
    r5 = BatchNormalization()(r4)
    r6 = Dense(classes, activation = 'softmax')(r5)
    
    model = Model(inputs = [input_hb_avg, input_hb_var, input_tf, input_hrv], outputs = r6)


    if(classes == 2) :
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['categorical_accuracy'])  
    else:
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['categorical_accuracy', f1])

    return model

def CNNModel2(classes, hb_length = 180,cutoff = 5000):
    # Create model
    act = 'sigmoid'
    act2 = 'sigmoid'
    d_loss = 0.55

    input_hb_avg = Input(shape = (hb_length,1))
    input_hb_var = Input(shape = (hb_length,1))
    input_tf     = Input(shape = (cutoff,1))

    x0 = GaussianNoise(30.0)(input_hb_avg)
    x = Conv1D(16, kernel_size = 50, activation = act)(x0)
    x1 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (x)
    x2 = Conv1D(32, kernel_size= 25, activation=act)(x1)
    x3 = MaxPooling1D(pool_size=3, strides=2)(x2)
    x4 = Model(inputs = input_hb_avg, outputs = x3 )

    y0 = GaussianNoise(30.0)(input_hb_var)
    y = Conv1D(16, kernel_size = 50, activation = act)(y0)
    y1 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (y)
    y2 = Conv1D(32, kernel_size= 25, activation=act)(y1)
    y3 = MaxPooling1D(pool_size=3, strides=2)(y2)
    y4 = Model(inputs = input_hb_var, outputs = y3 )

    combined = Concatenate()( [x4.output, y4.output])

    z = Conv1D(32, kernel_size= 20, activation=act)(combined)
    z1 = MaxPooling1D(pool_size=3, strides=2)(z)
    z2 = Model(inputs = [input_hb_avg, input_hb_var], outputs = z1)

    q0 = GaussianNoise(200.0)(input_tf)
    q = Conv1D(16, kernel_size = 200, activation = act)(q0)
    q1 = MaxPooling1D(pool_size=8, strides=6)(q)
    q2 = Conv1D(32, kernel_size= 150, activation=act)(q1)
    q3 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (q2)
    q30 = Conv1D(32, kernel_size= 100, activation=act)(q3)
    q31 = MaxPooling1D(pool_size=4, strides=3)(q30)
    q4 = Flatten()(q31)
    q5 = Dense(512, activation = act2)(q4)

    q5 = BatchNormalization()(q5)
    q6 = Reshape((16,32))(q5)
    q4 = Model(inputs = input_tf, outputs = q6 )

    bypass = Flatten()(q2)
    bypass = Dense(256, activation = act2)(bypass)
    bypass = Dropout(d_loss)(bypass)
    bypass = BatchNormalization()(bypass)


    combined_2 = Concatenate()([ z2.output, q4.output])

    r0 = Flatten() (combined_2)

    r = Dense(1024, activation = act2) (r0)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)

    r = Dense(512, activation = act2) (r)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)

    
    r = Dense(256, activation = act2) (r0)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)
    r = Add()([r, bypass])
    
    r3 = Dense(128, activation = act2) (r)
    r4 = Dropout(d_loss)(r3)
    r5 = BatchNormalization()(r4)
    r6 = Dense(classes, activation = 'softmax')(r5)
    
    model = Model(inputs = [input_hb_avg, input_hb_var, input_tf], outputs = r6)


    if(classes == 2) :
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['categorical_accuracy'])  
    else:
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['categorical_accuracy', f1])

    return model


def CNNModel1(classes, hb_length = 180,cutoff = 5000):
    # Create model
    act = 'relu'
    act2 = 'relu'
    d_loss = 0.55

    input_hb_avg = Input(shape = (hb_length,1))
    input_hb_var = Input(shape = (hb_length,1))
    input_tf     = Input(shape = (cutoff,1))

    x0 = GaussianNoise(10.0)(input_hb_avg)
    x = Conv1D(16, kernel_size = 50, activation = act)(x0)
    x1 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (x)
    x2 = Conv1D(32, kernel_size= 25, activation=act)(x1)
    x3 = MaxPooling1D(pool_size=3, strides=2)(x2)
    x4 = Model(inputs = input_hb_avg, outputs = x3 )

    y0 = GaussianNoise(10.0)(input_hb_var)
    y = Conv1D(16, kernel_size = 50, activation = act)(y0)
    y1 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (y)
    y2 = Conv1D(32, kernel_size= 25, activation=act)(y1)
    y3 = MaxPooling1D(pool_size=3, strides=2)(y2)
    y4 = Model(inputs = input_hb_var, outputs = y3 )

    combined = Concatenate()( [x4.output, y4.output])

    z = Conv1D(32, kernel_size= 20, activation=act)(combined)
    z1 = MaxPooling1D(pool_size=3, strides=2)(z)
    z2 = Model(inputs = [input_hb_avg, input_hb_var], outputs = z1)

    q0 = GaussianNoise(50.0)(input_tf)
    q = Conv1D(16, kernel_size = 200, activation = act)(q0)
    q1 = MaxPooling1D(pool_size=8, strides=6)(q)
    q2 = Conv1D(32, kernel_size= 150, activation=act)(q1)
    q3 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (q2)
    q30 = Conv1D(32, kernel_size= 100, activation=act)(q3)
    q31 = MaxPooling1D(pool_size=4, strides=3)(q30)
    q4 = Flatten()(q31)
    q5 = Dense(512, activation = act2)(q4)

    q5 = BatchNormalization()(q5)
    q6 = Reshape((16,32))(q5)
    q4 = Model(inputs = input_tf, outputs = q6 )

    combined_2 = Concatenate()([ z2.output, q4.output])

    r0 = Flatten() (combined_2)

    r = Dense(1024, activation = act2) (r0)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)

    r = Dense(512, activation = act2) (r)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)

    
    r = Dense(256, activation = act2) (r0)
    r = Dropout(d_loss)(r)
    r = BatchNormalization()(r)
    
    r3 = Dense(128, activation = act2) (r)
    r4 = Dropout(d_loss)(r3)
    r5 = BatchNormalization()(r4)
    r6 = Dense(classes, activation = 'softmax')(r5)
    
    model = Model(inputs = [input_hb_avg, input_hb_var, input_tf], outputs = r6)


    if(classes == 2) :
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['categorical_accuracy'])  
    else:
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['categorical_accuracy'])

    return model


def CNNModel(classes, hb_length = 180,cutoff = 5000):
    # Create model
    act = 'relu'
    d_loss = 0.45

    input_hb_avg = Input(shape = (hb_length,1))
    input_hb_var = Input(shape = (hb_length,1))
    input_tf     = Input(shape = (cutoff,1))

    x = Conv1D(32, kernel_size = 50, activation = act)(input_hb_avg)
    x1 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (x)
    x2 = Conv1D(64, kernel_size= 20, activation=act)(x1)
    x3 = MaxPooling1D(pool_size=3, strides=2)(x2)
    x4 = Model(inputs = input_hb_avg, outputs = x3 )

    y = Conv1D(32, kernel_size = 50, activation = act)(input_hb_var)
    y1 = Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32)) (y)
    y2 = Conv1D(64, kernel_size= 20, activation=act)(y1)
    y3 = MaxPooling1D(pool_size=3, strides=2)(y2)
    y4 = Model(inputs = input_hb_var, outputs = y3 )

    combined = Concatenate()( [x4.output, y4.output])

    z = Conv1D(128, kernel_size= 20, activation=act)(combined)
    z1 = MaxPooling1D(pool_size=3, strides=2)(z)
    z2 = Model(inputs = [input_hb_avg, input_hb_var], outputs = z1)

    q = Conv1D(32, kernel_size = 200, activation = act)(input_tf)
    q1 = MaxPooling1D(pool_size=4, strides=3)(q)
    q2 = Conv1D(64, kernel_size= 10, activation=act)(q1)
    q3 = MaxPooling1D(pool_size=3, strides=2)(q2)
    q4 = Flatten()(q3)
    q5 = Dense(2176, activation = act)(q4)
    q6 = Reshape((17,128))(q5)
    q4 = Model(inputs = input_tf, outputs = q6 )

    combined_2 = Concatenate()([ z2.output, q4.output])

    r0 = Flatten() (combined_2)
    r = Dense(512, activation = act) (r0)
    r1 = Dropout(d_loss)(r)
    r2 = BatchNormalization()(r1)
    r3 = Dense(256, activation = act) (r2)
    r4 = Dropout(d_loss)(r3)
    r5 = BatchNormalization()(r4)
    r6 = Dense(classes, activation = 'softmax')(r5)
    
    model = Model(inputs = [input_hb_avg, input_hb_var, input_tf], outputs = r6)

    if(classes == 2) :
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['categorical_accuracy'])  
    else:
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['categorical_accuracy'])

    return model

def surrogate(y_train):
    y_one = []
    for i in y_train:
        if i == 3:
            y_one.append(1)
        else :
            y_one.append(0)
    return y_one
    
    

def label(y_pred) :
    y_labeled = []
    for i in y_pred:
        y_labeled.append(int(np.where(i == np.amax(i))[0]))
    return y_labeled

def clean_c3tr (train_x_hb, train_x_tf, train_y, c_ = 3):
    bounds = train_y.size
    i = 0
    while i < bounds :
        if train_y[i] == c_:
            train_x_hb = np.delete(train_x_hb, i, 0)
            train_x_tf = np.delete(train_x_tf, i, 0)
            train_y = np.delete(train_y,i)
            bounds -= 1
        else :
            i += 1
    return train_x_hb, train_x_tf, train_y

def clean_c3te (train_x_hb, train_x_tf, train_y):
    bounds = len(train_y)
    indices = np.arange(0, bounds)
    i = 0
    while i < bounds :
        if train_y[i] == 1:
            train_x_hb = np.delete(train_x_hb, i, 0)
            train_x_tf = np.delete(train_x_tf, i, 0)
            train_y = np.delete(train_y,i)
            indices = np.delete(indices, i)
            bounds -= 1
            #print('one c3 occurrence deleted')
        else :
            i += 1
    return train_x_hb, train_x_tf, indices


def stitch_together(y_s_pred, y_pred, index):
    y_full_pred = np.full((len(y_s_pred)), -1)
    for i in range(0, len(y_s_pred)) :
        if y_s_pred[i] == 1 :
            y_full_pred[i] = 3
    count = 0
    for i in index:
        y_full_pred[i] = y_pred[count]
        count +=1
    return y_full_pred



def CNN_predict(train_x_hb, train_x_tf, train_y, test_x_hb, test_x_tf, eoe = False) :

    predictions = np.size(test_x_tf,0)
    #segment dataset into c3 vs rest
    #weigh classes
    X_train_hb, X_test_hb, X_train_tf, X_test_tf,  y_train, y_test = train_test_split(train_x_hb, train_x_tf, train_y, test_size=0.04, stratify = train_y)


    y_s = surrogate(y_train)
    sample_weights = get_class_weights(y_s)

    #train model on class 3 vs. rest
    y_train_oneh_1 = np_utils.to_categorical(y_s)
    
    cnn1 = CNNModel1(2, X_train_hb[0,0].size, test_x_tf[0].size)
   

    #evaluate performance with nicely splitted dtaset
    
    oldscore = 0
    bestweights = cnn1.get_weights()

    for i in range(0,500):
        cnn1.fit([X_train_hb[:,0], X_train_hb[:,1], X_train_tf], y_train_oneh_1, verbose = 1, epochs = 5, batch_size = 128, class_weight = sample_weights)
        y_pred_t = label(cnn1.predict([X_test_hb[:,0], X_test_hb[:,1], X_test_tf]))
        score = f1_score(surrogate(y_test), y_pred_t, average='micro')

        if score > oldscore :
            bestweights = cnn1.get_weights()
            oldscore = score
            print('new best score is: ', oldscore)
    cnn1.set_weights(bestweights)
    cnn1.save_weights('weights_c3.csv')
    
    cnn1.load_weights('weights_c3.csv')
    y_s_pred = label(cnn1.predict([test_x_hb[:,0], test_x_hb[:,1], test_x_tf]))

    #remove all c3 instances from test and train dataset, BUT FIRST LABEL BY ID!
    train_x_hb, train_x_tf, train_y = clean_c3tr(train_x_hb, train_x_tf, train_y)
    test_x_hb, test_x_tf, index = clean_c3te(test_x_hb, test_x_tf, y_s_pred)

    del cnn1
    for i in range(4):
        gc.collect()

    #reset class weights
    
    sample_weights = get_class_weights(train_y)

    #split Dataset again

    X_train_hb, X_test_hb, X_train_tf, X_test_tf,  y_train, y_test = train_test_split(train_x_hb, train_x_tf, train_y, test_size=0.04, stratify = train_y)


    cnn2 = CNNModel2(3, X_train_hb[0,0].size, test_x_tf[0].size)
    
    y_train_oneh_2 = np_utils.to_categorical(y_train)

    oldscore = 0
    bestweights = cnn2.get_weights()

    #train model on c1, c2, c3

    for i in range(0,2500):
        cnn2.fit([X_train_hb[:,0], X_train_hb[:,1], X_train_tf], y_train_oneh_2, verbose = 1, epochs = 1, batch_size = 128, class_weight = sample_weights)
        y_pred_t = label(cnn2.predict([X_test_hb[:,0], X_test_hb[:,1], X_test_tf]))
        score = f1_score(y_test, y_pred_t, average='micro')
        print('validation score of ', score, ' at epoch ', i)

        if score > oldscore :
            bestweights = cnn2.get_weights()
            cnn2.save_weights('weights_scored_' + str(score) +'.csv')
            oldscore = score
            print('new best score is: ', oldscore)
    if eoe == False :
        cnn2.set_weights(bestweights)
    
    y_pred = label(cnn2.predict([test_x_hb[:,0], test_x_hb[:,1], test_x_tf]))

    y_full_pred = stitch_together(y_s_pred, y_pred, index)

    return y_full_pred


