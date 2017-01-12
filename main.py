import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nxgboost import NXGBoost
from sklearn.ensemble import RandomForestRegressor

def load_data(name):
    with open('data/' + name + '.txt', 'r') as f:
        x = []
        for line in f:
            line = map(np.float32, line.strip().split('\t'))
            x.append(line)
        x = np.array(x)

    with open('data/' + name + 'label.txt', 'r') as f:
        y = []
        for line in f:
            line = map(np.float32, line.strip().split('\t'))
            y.append(line)
        y = np.array(y) - 1

    data = np.concatenate((x, y), axis=1)
    np.random.shuffle(data)

    n_test = len(data) / 3

    train_x = data[: -n_test, :-1]
    train_y = data[: -n_test, -1:]

    test_x = data[-n_test: , :-1]
    test_y = data[-n_test: , -1:]

    return train_x, train_y, test_x, test_y

def nxgb_callback(train_x, train_y, test_x):
    nxgb = NXGBoost()
    nxgb.fit(train_x, train_y, n_estimators=50, eta=0.1, lambd=1e-2, max_depth=5)
    return nxgb.predict(test_x)

def rf_callback(train_x, train_y, test_x):
    rf = RandomForestRegressor(n_estimators=50, max_depth=5)
    rf.fit(train_x, train_y.ravel())
    return rf.predict(test_x)

def run_model(callback, train_x, train_y, test_x, test_y):
    test_y_ = []
    for i in xrange(int(train_y.max()) + 1):
        test_y_.append(callback(train_x, (train_y == i).astype(np.float32), test_x).reshape([-1, 1]))
    test_y_ = np.argmax(test_y_, axis=0)
    return np.mean(np.equal(test_y_, test_y).astype(np.float32))

def run(name):
    data = load_data(name)
    nxgb_acc = run_model(nxgb_callback, *data)
    rf_acc = run_model(rf_callback, *data)
    return nxgb_acc, rf_acc

if __name__ == '__main__':
    np.random.seed(7)
    dataset_names = set()
    result = []
    for file in os.listdir('data'):
        filename = file.split('.txt')[0]
        dataset_names.add(filename.split('label')[0])

    i = 0
    for dataset_name in dataset_names:
        i += 1
        print '[' + str(i) + '/' + str(len(dataset_names)) + '] Calculating dataset', dataset_name, '...'
        
        nxgb_acc, rf_acc = run(dataset_name)
        print 'nxgboost acc:\t', nxgb_acc
        print 'random foreset acc:\t', rf_acc
        print ''

        result.append(('nxgboost', dataset_name, nxgb_acc))
        result.append(('random forest', dataset_name, rf_acc))

    print 'Calculation Finished.'

    result = pd.DataFrame(result, columns=['model', 'dataset', 'accuracy'])
    sns.set(style="whitegrid")
    g = sns.factorplot("dataset", "accuracy", "model", 
                    data=result, saturation=5, 
                    size=4, aspect=3, 
                    kind="bar", legend=False)
    plt.legend(loc='upper right')
    plt.savefig('fig/result.png')
    print 'Result is saved into the /fig folder'

    plt.show()