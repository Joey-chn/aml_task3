import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.decomposition import PCA, KernelPCA


def standarlization(train_x, test_x):
    # standarlization
    scalar = StandardScaler()
    train_x = scalar.fit_transform(train_x.astype('float64'))
    test_x = scalar.transform(test_x.astype('float64'))
    return train_x, test_x


def svmClassifier(train_x, train_y, test_x):
    train_y = train_y.ravel()
    class_weighting = {0: 1.6, 1: 0.2, 2: 1.6}

    classifier = SVC(class_weight=class_weighting, gamma=0.0003, C=1)  # c the penalty term for misclassification
    # make balanced_accuracy_scorer
    score_func = make_scorer(balanced_accuracy_score)
    # cross validation
    scores = cross_val_score(classifier, train_x, train_y, cv=5, scoring=score_func)
    print(scores)

    # learn on all data
    classifier.fit(train_x, train_y)
    y_predict_test = classifier.predict(test_x)
    return y_predict_test


def grid_search(train_x, train_y, test_x):
    parameters = {'C': [0.8, 1, 1.2], 'gamma': [0.0003, 0.0005, 0.001]}
    svcClassifier = SVC(kernel='rbf', class_weight='balanced')
    score_func = make_scorer(balanced_accuracy_score)
    gs = GridSearchCV(svcClassifier, parameters, cv=5, scoring=score_func)
    gs.fit(train_x, train_y)
    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)
    y_predict_test = gs.predict(test_x)
    return y_predict_test


def pincipal_component(train_x, test_x, n_com):
    pca = PCA(n_components=n_com)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    return train_x, test_x

def kernel_pincipal_component(train_x, test_x, n_com, kernel_type = 'rbf'):
    pca = KernelPCA(n_components=n_com, kernel= kernel_type)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    return train_x, test_x


def read_from_file(X_train_file, y_train_file, X_predict_file):
    # read from files
    x_train = pd.read_csv(X_train_file)  # 1212*833
    y_train = pd.read_csv(y_train_file, index_col='id')  # 1212*1
    xy = np.concatenate([x_train, y_train], axis=1)
    print(xy.shape)
    np.random.shuffle(xy)
    x_train = xy[:, :-1]
    y_train = xy[:, -1]
    print(x_train.shape, y_train.shape)
    x_predict = pd.read_csv(X_predict_file)
    return x_train, y_train, x_predict


def oversampling_data(x_train, y_train):
    handler = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = handler.fit_resample(x_train, y_train)
    return X_resampled, y_resampled


def feature_selection(x_train, y_train, x_predict, n_feature):
    # apply SelectKBest class to extract top 10 best features
    selector = SelectKBest(score_func=f_classif, k=n_feature)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_predict_selected = selector.transform(x_predict)
    # scores = selector.scores_
    # scores.sort()
    # scores = scores[::-1]
    # idx = 1
    # for i in scores:
    #     print("{:} : {:} ".format(idx, i))
    #     idx += 1
    return x_train_selected, x_predict_selected


def result_to_csv(predict_y, sample_file):
    # write the result to the CSV file
    sample_file = pd.read_csv(sample_file)
    id = sample_file['id'].to_numpy().reshape(-1, 1)
    result = np.concatenate((id, predict_y.reshape(-1, 1)), axis=1)
    result = pd.DataFrame(result, columns=['id', 'y'])
    result.to_csv('predict_y.csv', index=False)


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

if __name__ == '__main__':
    is_resampled = False
    # read from file check the size of the file
    all_data = read_from_file("X_train.csv", "y_train.csv", "X_test.csv")
    x_train = all_data[0]
    y_train = all_data[1]
    x_test = all_data[2]
    # print("x_train {:}, y_train{:}, x_test{:}".format(np.shape(x_train), np.shape(y_train), np.shape(x_test)))

    if is_resampled:
        resampled_data = oversampling_data(x_train, y_train)
        x_train = resampled_data[0]
        y_train = resampled_data[1]

    # check_balance(y_train)

    # standarlization
    x_std = standarlization(x_train, x_test)
    x_train = x_std[0]
    x_test = x_std[1]

    # feature_selection
    selected_data = feature_selection(x_train, y_train, x_test, n_feature=800)
    x_train_selected = selected_data[0]
    x_test_selected = selected_data[1]

    # principal component, linear doesn't make sense here

    # check data balancing
    check_balance(y_train)

    # prediction
    y_predict = grid_search(x_train_selected, y_train, x_test_selected)
    y_predict = svmClassifier(x_train_selected, y_train, x_test_selected)
    result_to_csv(y_predict, 'sample.csv')