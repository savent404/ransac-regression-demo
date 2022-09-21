from math import fmod
import os
from statistics import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


def load_data(use_cache):
    if use_cache is False:
        raw_data = np.loadtxt(open('1.csv', 'rb'),
                              delimiter=';', skiprows=1, dtype='str')
        data = []
        angle_list = []
        sector_list = []

        for i in range(0, raw_data.shape[0]):
            obj = raw_data[i, :]

            angle = eval(obj[1])
            sector = eval(obj[2])
            valid = eval(obj[3])
            if valid > 0 and sector > -400:
                angle_list.append(angle)
                sector_list.append(sector)

        assert (len(angle_list) == len(sector_list))

        data_1 = np.array(angle_list)
        data_2 = np.array(sector_list)
        data = np.column_stack([data_1, data_2])
        np.save('.cache.npy', data)
    else:
        data = np.load('.cache.npy')

    return data


def print_seq_figure(data):
    '''
    按照sector打印变化曲线
    '''
    _data = data[np.argsort(data[:, 1])]
    _x = _data[:, 1]
    _y = _data[:, 0]
    fig, ax = plt.subplots()
    fig.legend('x = sector')
    ax.scatter(_x, _y)


def data_filter(data):
    offset = 26
    list = []

    for i in range(data.shape[0]):
        y = data[i, 0]
        x = data[i, 1]

        if x == 0 and y == 0:
            continue

        x = fmod(x + offset, 60)

        if x < 0:
            x = x + 60

        x = x / 60.0
        y = y / 1024.0
        list.append((x, y))
    _data = np.array(list)
    _data = _data[np.argsort(_data[:, 0])]
    return _data


def print_sector_xy(data):
    _data = data
    _x = _data[:, 1]
    _y = _data[:, 0]
    fig, ax = plt.subplots()
    fig.legend('x=sector%60')
    ax.scatter(_x, _y)


def curv_fund(x, p0, p1, p2, p3):
    return np.power(x, 3) * p3 + np.power(x, 2) * p2 + np.power(x, 1) * p1 + p0


def curv_fit(data):
    x = data[:, 0]
    y = data[:, 1]
    popt, pcov = curve_fit(curv_fund, x, y)

    fx = np.arange(0, 1, 0.001)
    fy = [curv_fund(i, popt[0], popt[1], popt[2], popt[3]) for i in fx]
    print('curv fit opt:', popt)
    return fx, fy


def ransac_fit(data):
    x = data[:, 0]
    y = data[:, 1]
    row = x.shape[0]

    train_data, test_data = train_test_split(data, train_size=0.8)

    train_x, train_y = train_data[:, 0], train_data[:, 1]
    test_x, test_y = test_data[:, 0], test_data[:, 1]

    reg = PolynomialFeatures(degree=3)

    train_x_poly = reg.fit_transform(train_x[:, np.newaxis])
    test_x_poly = reg.fit_transform(test_x[:, np.newaxis])

    model = RANSACRegressor()
    m = model.fit(train_x_poly, train_y)
    score = model.score(test_x_poly, test_y)
    _x = np.arange(0, 1, 0.01)
    _x_poly = reg.fit_transform(_x[:, np.newaxis])

    # 由于是多项式方程，直接将intercept_叠加在coef_[0]上
    arg = m.estimator_.coef_
    arg[0] = arg[0] + m.estimator_.intercept_
    print('score:', score, 'arg:', arg)
    #return _x, model.predict(_x_poly)
    #return _x, _x_poly.dot(arg) + m.estimator_.intercept_
    return _x, _x_poly.dot(arg)


def plot_fit_result(data, fit_res):
    fig, ax = plt.subplots()
    fig.legend('fit plot')

    x = data[:, 0]
    y = data[:, 1]
    ax.plot(x, y, 'b.')
    ax.legend('original')

    wx = np.arange(0, 1, 0.001)
    wy = wx
    ax.plot(wx, wy, 'y--')
    ax.legend(b'wanted')

    fx = fit_res[0]
    fy = fit_res[1]
    ax.plot(fx, fy, 'r--')
    ax.legend('after')


has_cache = os.path.exists('.cache.npy')
data = load_data(has_cache)

filtered_data = data_filter(data)

print_seq_figure(data)
res1 = curv_fit(filtered_data)
res2 = ransac_fit(filtered_data)
plot_fit_result(filtered_data, res1)
plot_fit_result(filtered_data, res2)

print('adj arg(anti-func of ransac fit):')
#anti_data = np.array(filtered_data.shape)
anti_data = filtered_data
anti_data = filtered_data[:, [1,0]]
adj = ransac_fit(anti_data)
plot_fit_result(filtered_data, adj)
plt.show()
