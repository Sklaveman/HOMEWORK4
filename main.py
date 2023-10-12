#!/home/artux/Documents/anaconda3/bin/python3

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from useful_package import hyperbola, polynom_3


for name, func in zip(["hyperbola", "polynom_3"], [hyperbola, polynom_3]):
    train_size, test_size = 1000, 100
    X_train = 1 + np.random.rand(train_size).reshape(-1, 1)
    Y_train = func(X_train)
    X_test = 1 + np.random.rand(test_size).reshape(-1, 1)
    Y_test = func(X_test)

    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    mse = np.sum((model.predict(X_test) - Y_test)**2)
    print(name, mse)
