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


def read_from_file(X_train_file, y_train_file, X_predict_file):
    # read from files
    x_train = pd.read_csv(X_train_file, index_col='id').to_numpy()
    y_train = pd.read_csv(y_train_file, index_col='id').to_numpy()
    # convert it to np array

    # xy = np.concatenate([x_train, y_train], axis=1)
    # np.random.shuffle(xy)
    # x_train = xy[:, :-1]
    # y_train = xy[:, -1]
    # print(x_train.shape, y_train.shape) # num of points : 5117, rang of ecg 17814
    x_predict = pd.read_csv(X_predict_file).to_numpy()
    return x_train, y_train, x_predict


def feature_extraction(X):
    # get all the templates for one person, take the median value, get one template for each person
    # remove nan value in nparray
    X_new = []
    for row in X:
        row = row[np.logical_not(np.isnan(row))]
        # extract all heartbeats templates
        signal_processed = ecg.ecg(signal=row, sampling_rate=300, show=False)
        templates = signal_processed[4]
        # take the median of templates along row dimension
        template_median = np.median(templates, axis=0)
        # take the minimum R peaks
        rpeaks_location = signal_processed[2]
        rpeaks = row[rpeaks_location]
        rpeaks_min = min(rpeaks)
        rpeaks_max = max(rpeaks)
        features = np.append(template_median, [rpeaks_min, rpeaks_max])
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


def check_balance(train_y):
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    train_y = train_y.squeeze()
    for elem in train_y:
        if elem == 0:
            count_0 += 1
        elif elem == 1:
            count_1 += 1
        elif elem == 2:
            count_2 += 1
        else:
            count_3 += 1
    print("class 0: {:}, class 1: {:}, class 2: {:}, class 3: {:}".format(count_0, count_1, count_2, count_3))
    # class 0: 3030, class 1: 443, class 2: 1474, class 3: 170


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
    class_weighting = {0: 1.6, 1: 0.2, 2: 1.6}

    classifier = SVC(class_weight='balanced', gamma=0.1, C=5)  # c the penalty term for misclassification
    # make balanced_accuracy_scorer
    score_func = make_scorer(f1_score, average='micro') # additional param for f1_score
    # cross validation
    scores = cross_val_score(classifier, train_x, train_y, cv=5, scoring=score_func)
    print(scores)

    # learn on all data
    classifier.fit(train_x, train_y)
    y_predict_test = classifier.predict(test_x)
    return y_predict_test


if __name__ == '__main__':
    is_start = False
    # read data from files
    if is_start:
        all_data = read_from_file("X_train.csv", "y_train.csv", "X_test.csv")
        y_train = all_data[1]
        x_train_raw = all_data[0]
        x_test_raw = all_data[2]

        print(x_train_raw.shape, y_train.shape)
        # feature extraction for x_train and x_test
        x_train_temMed = feature_extraction(x_train_raw) # 180 features
        x_test_temMed =  feature_extraction(x_test_raw) # 180 features

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
        print(x_train_std[[10, 14, 17, 18]][:, -2:])
    # prediction
    # y_predict = grid_search(x_train_selected, y_train, x_test_selected)
    y_predict = svmClassifier(x_train_std, y_train, x_test_std)
    # neural net

    result_to_csv(y_predict, 'sample.csv')


