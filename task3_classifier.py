import pandas
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif
from keras.utils import np_utils
#from neuralNet_OG import simpleClassify, svmClassify, rfClassify, bgmClassify, label, svcClassify
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import csv
from biosppy.signals import ecg
import matplotlib.pyplot as plt


def open_from_csv(X_train_file, y_train_file, X_predict_file) : 
    # read from files
    x_train   = pandas.read_csv(X_train_file,   index_col='id') # 1212*833
    y_train   = pandas.read_csv(y_train_file,   index_col='id') # 1212*1
    x_predict = pandas.read_csv(X_predict_file, index_col='id')
    return x_train.values, y_train.values, x_predict.values

def get_class_weights(y_train) :
    y_integers = np.argmax(np_utils.to_categorical(y_train), axis=1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    return dict(enumerate(class_weights))
    #return class_weights

def feature_selection(X_train, y_train, x_predict, variables = 500):

    # apply SelectKBest class to extract best features
    selector = SelectKBest(score_func = f_classif, k=variables) 
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_predict_selected = selector.transform(x_predict)
    print(x_train_selected.shape)

    return x_train_selected, x_predict_selected, selector


def result_to_csv(predict_y):
    # write the result to the CSV file
    id = np.arange(np.size(y_pred)).reshape(-1,1)
    result = np.concatenate((id, predict_y.reshape(-1,1)), axis=1)
    result = pandas.DataFrame(result, columns=['id', 'y'])
    result.to_csv('predict_y.csv', index=False)


def find_no_relevant_variables_CV(x_train, y_train, sample_weights, skip = False) :
    if skip : #best result at 270
        return 480
    else :
        basevar = 300
        increment = 20
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        scores = np.zeros(10)
        for train, test in kfold.split(x_train, y_train) :
            for i in range(0,10) :
                pca = PCA(n_components = basevar + i*increment)
                x_train_selected = pca.fit_transform(x_train[train])
                scores[i] += svcClassify(x_train_selected, y_train[train],
                    pca.transform(x_train[test]), y_train[test], sample_weights,
                    _C = .38)

        result_num_var = pandas.DataFrame(scores.reshape(-1,1), columns = ['scores'])
        result_num_var.to_csv('relevant_vars_score_' + str(basevar) + '_' + str(increment) + '.csv')

        idx = int(np.where(scores == np.amax(scores))[0])
        print('best estimate for number of relevant variables found at ', basevar + idx*increment)
        return basevar + idx*increment

def find_best_params_SVM(x_train, y_train, sample_weights, skip = False) :
    if skip :
        return 0.74 #found by CV: .72
    Cbase = 0.72
    Cstep = .01
    steps = 10
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    scores = np.zeros(steps)
    for train, test in kfold.split(x_train, y_train) :
        for i in range(0,steps) :
            scores[i] += svcClassify(x_train[train], y_train[train],
                x_train[test], y_train[test], sample_weights,
                kernel = 'rbf', _C = Cbase + i*Cstep, gamma = 'scale')

    c_s = np.linspace(Cbase,Cbase + (steps-1)*Cstep,steps).reshape(-1,1)
    result_num_var = pandas.DataFrame( np.concatenate((c_s, scores.reshape(-1,1)), 1), columns = ['C','scores'])
    result_num_var.to_csv('svc_params_score_'+str(Cbase)+'C_' + str(Cstep)+'_10x.csv')

    idx = int(np.where(scores == np.amax(scores))[0])
    print('best estimate for C ', Cbase + idx*Cstep)
    return Cbase + idx*Cstep

def find_best_gamma_SVM(x_train, y_train, sample_weights, skip = False, c_ = .72) :
    if skip :
        return 0.0004 #best param found by CV: 0.0007
    gbase = 0.0003
    gstep = 0.0001
    steps = 10
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    scores = np.zeros(steps)
    for train, test in kfold.split(x_train, y_train) :
        for i in range(0,steps) :
            scores[i] += svcClassify(x_train[train], y_train[train],
                x_train[test], y_train[test], sample_weights,
                kernel = 'rbf', _C = c_, gamma = gbase + i*gstep)

    c_s = np.linspace(gbase,gbase + (steps-1)*gstep,steps).reshape(-1,1)
    result_num_var = pandas.DataFrame( np.concatenate((c_s, scores.reshape(-1,1)), 1), columns = ['C','scores'])
    result_num_var.to_csv('svc_params_score_i'+str(gbase)+'gamma_' + str(gstep)+'_10x.csv')

    idx = int(np.where(scores == np.amax(scores))[0])
    print('best estimate for gamma ', gbase + idx*gstep)
    return gbase + idx*gstep


def find_best_params_RF(x_train, y_train, sample_weights, skip = False) :
    if skip :
        return 6
    esbase = 4
    esstep = 4
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    scores = np.zeros(10)
    for train, test in kfold.split(x_train, y_train) :
        for i in range(0,10) :
            scores[i] += rfClassify(x_train[train], y_train[train],
                x_train[test], y_train[test], sample_weights,
                estimators = esbase + i*esstep)

    result_num_var = pandas.DataFrame(scores.reshape(-1,1), columns = ['scores'])
    result_num_var.to_csv('rf_params_score_04es_4.csv')

    idx = int(np.where(scores == np.amax(scores))[0])
    print('best estimate for no of estimators ', esbase + esstep*idx)
    return  esbase + esstep*idx

def find_best_params_NN(x_train, y_train, sample_weights, skip = False) :
    if skip :
        return 1024
    sizebase = 64*8
    increment = 264    

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    scores = np.zeros(5)
    for train, test in kfold.split(x_train, y_train) :
        for i in range(0,5) :
            scores[i] += simpleClassify(x_train[train], y_train[train],
                x_train[test], y_train[test], sample_weights,
                neurons = sizebase + i*increment)

    result_num_var = pandas.DataFrame(scores.reshape(-1,1), columns = ['scores'])
    result_num_var.to_csv('NN_params_score_512n_264_pca.csv')

    idx = int(np.where(scores == np.amax(scores))[0])
    print('best estimate for no of estimators ', sizebase + idx*increment)
    return sizebase + idx*increment

def model_choice_cv(x_train, y_train, sample_weights, skip = False) :
    if skip :
        return 'NN'
    model_list = ['SVM', 'RF', 'SVMS', 'GP']
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    scores = np.zeros(4)
    for train, test in kfold.split(x_train, y_train) :
        scores[0] += svmClassify(x_train[train], y_train[train],
            x_train[test], y_train[test], sample_weights,
            kernel = 'rbf', nu = 0.06)
        scores[1] += rfClassify(x_train[train], y_train[train],
            x_train[test], y_train[test], sample_weights,
            estimators = 6)
        scores[2] += simpleClassify(x_train[train], y_train[train],
            x_train[test], y_train[test], sample_weights)
        scores[3] += svmClassify(x_train[train], y_train[train],
            x_train[test], y_train[test], sample_weights,
            kernel = 'linear', nu = 0.06)
#        scores[3] += bgmClassify(x_train[train], y_train[train],
#            x_train[test], y_train[test])
    
    result_num_var = pandas.DataFrame(scores.reshape(-1,1), columns = ['scores'])
    result_num_var.to_csv('models_scores.csv')
    idx = int(np.where(scores == np.amax(scores))[0])
    print('best performing model was a ', model_list[idx])
    return model_list[idx]

def ensemble_prediction(x_train, y_train, x_test, sample_weights):
    y_pred_svm = svmClassify(x_train, y_train, x_train, [],
        sample_weights, kernel = 'rbf', nu = 0.06, predict = True)
    print(y_pred_svm)
    y_pred_NN = simpleClassify(x_train, y_train, x_test, [],
        sample_weights, predict = True)
    y_true = []
    y_l_s = label(y_pred_svm)
    y_l_n = label(y_pred_NN)
    for i in range(0, np.size(x_test, 0)) :
        if y_l_s[i] == y_l_n[i]:
            y_true.append(y_l_s[i])
        else:
            prob_s = y_pred_svm[i, y_l_s[i]]
            prob_n = y_pred_NN[i, y_l_n[i]]
            if prob_n > prob_s :
                y_true.append(y_l_n[i])
            else:
                y_true.append(y_l_s[i])

    return np.asarray(y_true)
    

if __name__ == '__main__':
    #open files
    maxsize = 17813
#    x_train, y_train, x_test = open_from_csv('X_train.csv', 'y_train.csv', 'X_test.csv')

    y_train = pandas.read_csv( 'y_train.csv',   index_col='id').values
    with open('X_train.csv', 'r') as f:
        reader = csv.reader(f)
        vals = list(reader)[1:]
    [i.pop(0) for i in vals]


    #TODO: check if the spectral sigs differ across classes
    """
    spectrals = np.zeros((4 ,maxsize))
    cspectrals = np.zeros((4,maxsize), dtype = np.cdouble)
    speccount = np.zeros(4)
    count = 0

    for i in vals :
        idx = y_train[count]
        fft = np.fft.rfft(np.asarray(i, dtype='float64'), maxsize)
        spectrals[idx,:fft.size] = np.add(spectrals[idx,:fft.size] ,np.absolute(fft.reshape(1,-1)))
        #cspectrals[idx,:fft.size] = np.add(spectrals[idx,:fft.size] ,fft.reshape(1,-1))

        speccount[idx] += 1
        count += 1
    
    for i in range(4) :
        spectrals[i] /= speccount[i]
        #sns.lineplot( np.arange(0,maxsize), spectrals[i])
    sns.lineplot(np.arange(0,maxsize), np.subtract(spectrals[1], spectrals[2]))
    plt.show()
    """
    
    out = ecg.ecg(signal=np.asarray(vals[1], dtype='float64'), sampling_rate=300, show=True)
    
