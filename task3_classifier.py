import pandas
import numpy as np
import spectrum
from numpy import savetxt, loadtxt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import csv
from biosppy.signals import ecg, tools
import seaborn as sns
import matplotlib.pyplot as plt
from neuralNet import CNN_predict
import pyhrv as hr


def get_class_weights(y_train) :
    y_integers = np.argmax(np_utils.to_categorical(y_train), axis=1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    return dict(enumerate(class_weights))
    #return class_weights


def result_to_csv(predict_y):
    # write the result to the CSV file
    id = np.arange(np.size(predict_y)).reshape(-1,1)
    result = np.concatenate((id, predict_y.reshape(-1,1)), axis=1)
    result = pandas.DataFrame(result, columns=['id', 'y'])
    result.to_csv('predict_y.csv', index=False)


    
def read_data(x_trainame, y_trainame, x_testname) :
    y_train = pandas.read_csv( 'y_train.csv',   index_col='id').values
    with open(x_trainame, 'r') as f:
        reader = csv.reader(f)
        tr_vals = list(reader)[1:]
    [i.pop(0) for i in tr_vals]
    with open(x_testname, 'r') as f:
        reader = csv.reader(f)
        te_vals = list(reader)[1:]
    [i.pop(0) for i in te_vals]
    return tr_vals, y_train, te_vals

def convert_signals(train_sig, skip = False, fname = 'train_') :

    maxsize = 17813

    if skip :
        hb, tf = load_signals(fname)
        return hb, tf

    cutoff = 2500
    n = len(train_sig)

    train_x_tf = np.zeros((n,cutoff), dtype = 'float64')    

    train_x_hb = np.zeros((n, 2, 180), dtype = 'float64')   

    for i in range(0, n) :
        sig = np.asarray(train_sig[i], dtype = 'float64')
        train_x_tf[i] = tools.analytic_signal(signal = ecg.ecg(signal = sig, sampling_rate = 300, show = False)[1], N = cutoff, )[0]
        #train_x_tf[i] = np.absolute(np.fft.rfft(sig, n = maxsize))[:cutoff]
        rpeaks = ecg.christov_segmenter(signal = sig, sampling_rate = 300)
        out = ecg.extract_heartbeats(signal = sig, rpeaks = rpeaks[0], sampling_rate = 300)
        if out[0].size > 0 :
            train_x_hb[i,0]  = np.mean(out[0], axis = 0)
            train_x_hb[i,1] = np.std(out[0], axis = 0)
        else :
            train_x_hb[i,0] = np.zeros(180)
            train_x_hb[i,1] = np.zeros(180)
    
    return train_x_hb, train_x_tf 

def save_signals(hb, tf, signame) :
    savetxt(signame + 'hb.csv', hb.reshape(-1,180), delimiter=',')
    savetxt(signame + 'tf.csv', tf, delimiter=',')

def load_signals(signame) :
    hb = loadtxt(signame + 'hb.csv', delimiter=',')
    tf = loadtxt(signame + 'tf.csv', delimiter=',')
    return hb.reshape(-1, 2, 180, 1), tf.reshape(-1, tf[0].size, 1)

        
        
        
    

if __name__ == '__main__':
    #open files
    #read in signals
    train_sig, y_train, test_sig = read_data('X_train.csv', 'y_train.csv', 'X_test.csv' )

    data = hr.hrv(signal =np.asarray(train_sig[0], dtype = 'float64'), sampling_rate = 300.0)

    #convert to fixed-frame signals
    
    train_x_hb, train_x_tf, train_x_hr = convert_signals(train_sig)
    
    save_signals(train_x_hb, train_x_tf, 'train_')

    test_x_hb, test_x_tf, test_x_hr = convert_signals(test_sig)
    
    save_signals( test_x_hb, test_x_tf, 'test_')
    

    train_x_hb, train_x_tf = load_signals('train_')
    
    test_x_hb, test_x_tf = load_signals('test_')

    predict_y = CNN_predict(train_x_hb, train_x_tf, y_train, test_x_hb, test_x_tf)

    result_to_csv(predict_y)
    
    #DONE: check if the spectral sigs differ across classes
    #RESULT: significant differences between healthy & the two abnormal cases

