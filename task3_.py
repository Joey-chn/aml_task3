import biosppy.signals.ecg as ecg
import numpy as np
import pandas as pd

# from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.metrics import f1_score
# from neuralNet import neurNet_classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from pyhrv import hrv
import pyhrv.time_domain as td
from sklearn.model_selection import GridSearchCV
import neurokit as nk
from sklearn.externals import joblib



def read_from_file(X_train_file, y_train_file, X_predict_file, is_testing = True):
    x_predict = []
    y_train = []
    if is_testing:
        # read from files
        x_train = pd.read_csv(X_train_file, index_col='id', nrows = 30).to_numpy()
    else:
        x_train = pd.read_csv(X_train_file, index_col='id').to_numpy()
        y_train = pd.read_csv(y_train_file, index_col='id').to_numpy()
        x_predict = pd.read_csv(X_predict_file).to_numpy()
    return x_train, y_train, x_predict


def feature_extraction(X):
    # get all the templates for one person, take the median value, get one template for each person
    # remove nan value in nparray
    X_new = []
    count = 0
    for row in X:
        count += 1
        print(count)
        if count in [628, 629, 3501, 3721, 4702]:
            continue
        row = row[np.logical_not(np.isnan(row))]
        # extract all heartbeats templates
        signal_processed = ecg.ecg(signal=row, sampling_rate=300, show=False)
        templates = signal_processed[4]
        # take the median of templates along row dimension
        template_median = np.median(templates, axis=0)
        template_mean = np.mean(templates, axis = 0)
        # take the minimum R peaks
        rpeaks_location = signal_processed[2]
        rpeaks_location = ecg.correct_rpeaks(signal = row, rpeaks = rpeaks_location, sampling_rate=300)
        features_raw = nk.ecg_preprocess(ecg=row, sampling_rate=300)
        # Q-waves
        Q_idx = features_raw['ECG']['Q_Waves']
        # T_idx = features_raw['ECG']['T_Waves']
        Q_wave = row[Q_idx]
        # T_wave = row[T_idx]
        Q_min = min(Q_wave)
        Q_max = max(Q_wave)
        # if len(T_wave > 0):
        #     T_min = min(T_wave)
        #     T_max = max(T_wave)
        #     T_var = np.var(T_wave)
        # else:
        #     T_min, T_max, T_var = 0, 0, 0

        Q_var = np.var(Q_wave)
        # R-peaks
        rpeaks = row[rpeaks_location]
        rpeaks_min = min(rpeaks)
        rpeaks_max = max(rpeaks)
        rpeaks_mean = np.mean(rpeaks)
        rpeaks_var = np.var(rpeaks)
        # take the hearbeat rate
        # heartbeat_rate = signal_processed[-1]
        # in case where the heartbeat is empty in the result
        # if heartbeat_rate.size == 0:
        #     hb_rate_min = np.nan
        #     hb_rate_max = np.nan
        # else:
        #     hb_rate_min = min(heartbeat_rate)
        #     hb_rate_max = max(heartbeat_rate)
        # add RR intervals var into the feature
        rr_interval = np.diff(rpeaks_location)
        # print("rr_interval: ", rr_interval)
        rr_var = np.var(rr_interval)
        rr_min = np.min(rr_interval)
        rr_max = np.max(rr_interval)
        # add hrv into the feature
        # hrv_val = caculate_hrv(row, rpeaks)
        features = np.append(template_median, [rpeaks_min, rpeaks_max, rpeaks_mean, rpeaks_var, rr_min, rr_max, rr_var, Q_min, Q_max,  Q_var])
        # features = np.concatenate((template_mean, features), axis = 0)
        # add the new point into  all datapoints
        X_new.append(features)
    X_new = np.array(X_new)
    print(X_new.shape)
    return X_new


def processed_to_csv(X_train, flag = 'train'):
    X = np.asarray(X_train)
    if flag == 'test':
        np.savetxt('X_test_temMed.csv', X)
    else:
        np.savetxt('X_train_temMed.csv', X)


def result_to_csv(predict_y, sample_file):
    # write the result to the CSV file
    sample_file = pd.read_csv(sample_file)
    id = sample_file['id'].to_numpy().reshape(-1, 1)
    result = np.concatenate((id, predict_y.reshape(-1, 1)), axis=1)
    result = pd.DataFrame(result, columns=['id', 'y'])
    result.to_csv('predict_y.csv', index=False)


def standarlization(train_x, test_x):
    # standarlization
    scalar = StandardScaler()
    train_x = scalar.fit_transform(train_x.astype('float64'))
    test_x = scalar.transform(test_x.astype('float64'))
    return train_x, test_x


def svmClassifier(train_x, train_y, test_x):
    train_y = train_y.ravel()
    classifier = SVC(class_weight='balanced', gamma=0.02, C=10)  # c the penalty term for misclassification
    # make balanced_accuracy_scorer
    score_func = make_scorer(f1_score, average='micro') # additional param for f1_score
    # cross validation
    scores = cross_val_score(classifier, train_x, train_y, cv=5, scoring=score_func)
    print(scores)
    # learn on all data
    classifier.fit(train_x, train_y)
    y_predict_test = classifier.predict(test_x)
    return y_predict_test


def grid_search(train_x, train_y, test_x):
    parameters = {'C': [0.5, 1, 5, 10], 'gamma': [0.005, 0.01, 0.02, 0.05, 0.1]}
    svcClassifier = SVC(kernel='rbf', class_weight='balanced')
    score_func = make_scorer(f1_score, average='micro')
    gs = GridSearchCV(svcClassifier, parameters, cv=5, scoring=score_func)
    gs.fit(train_x, train_y)
    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)
    y_predict_test = gs.predict(test_x)
    return y_predict_test


def adaBoostClassifier(train_x, train_y, test_x):
    train_y = train_y.ravel()
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'), n_estimators=50, learning_rate=1)
    # make balanced_accuracy_scorer
    score_func = make_scorer(f1_score, average='micro')  # additional param for f1_score
    # cross validation
    scores = cross_val_score(classifier, train_x, train_y, cv=5, scoring=score_func)
    print(scores)
    # learn on all data
    classifier.fit(train_x, train_y)
    y_predict_test = classifier.predict(test_x)
    return y_predict_test


if __name__ == '__main__':
    is_start = True
    is_testing = False
    # read data from files
    if is_start:
        all_data = read_from_file("X_train.csv", "y_train.csv", "X_test.csv", is_testing)
        y_train = all_data[1]
        for i in [628, 629, 3501, 3721, 4702]:
            y_train = np.delete(y_train, i) # empty heartrate
        x_train_raw = all_data[0]
        x_test_raw = all_data[2]

        # feature extraction for x_train and x_test
        x_train_temMed = feature_extraction(x_train_raw)
        x_test_temMed =  feature_extraction(x_test_raw)

        # standarlization
        x_std = standarlization(x_train_temMed, x_test_temMed)
        x_train_std = x_std[0]
        x_test_std = x_std[1]
        # write processed data to csv
        processed_to_csv(x_train_std)
        processed_to_csv(x_test_std,flag = 'test')

    if not is_start:
        x_train_std =  pd.read_csv('X_train_temMed.csv', delimiter=' ', index_col=False, header = None).to_numpy()
        x_test_std = pd.read_csv('X_test_temMed.csv', delimiter=' ', index_col=False, header=None).to_numpy()
        y_train = pd.read_csv('y_train.csv', index_col='id').to_numpy()
        # print(x_train_std[[10, 14, 17, 18]][:, -2:])
    # prediction
    # y_predict = grid_search(x_train_std, y_train, x_test_std)
    y_predict = svmClassifier(x_train_std, y_train, x_test_std)
    # neural net
    # y_predict = neurNet_classifier(x_train_std, y_train, x_test_std)
    # Adaboost classifier
    # y_predict = adaBoostClassifier(x_train_std, y_train, x_test_std)
    # grid search
    result_to_csv(y_predict, 'sample.csv')


