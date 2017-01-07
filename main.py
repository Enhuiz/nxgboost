import numpy as np
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

    n_test = len(data) / 5

    train_x = data[: -n_test, :-1]
    train_y = data[: -n_test, -1:]

    test_x = data[-n_test: , :-1]
    test_y = data[-n_test: , -1:]

    return train_x, train_y, test_x, test_y

def nxgb_callback(train_x, train_y, test_x):
    nxgb = NXGBoost()
    nxgb.fit(train_x, train_y, n_estimators=5, lambd=1e-4, max_depth=5)
    return nxgb.predict(test_x)

def rf_callback(train_x, train_y, test_x):
    rf = RandomForestRegressor(n_estimators=5)
    rf.fit(train_x, train_y.ravel())
    return rf.predict(test_x)

def run(callback, train_x, train_y, test_x, test_y):
    test_y_ = []
    for i in xrange(int(train_y.max()) + 1):
        test_y_.append(callback(train_x, (train_y == i).astype(np.float32), test_x).reshape([-1, 1]))
    test_y_ = np.argmax(test_y_, axis=0)
    return np.mean(np.equal(test_y_, test_y).astype(np.float32))

if __name__ == '__main__':
    np.random.seed(27)
    data = load_data('wdbc')
    print 'nxgb accuracy:', run(nxgb_callback, *data)
    print 'random forest accuracy:', run(rf_callback, *data)