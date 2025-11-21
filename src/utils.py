def error_metric(y, y_hat, c):
    import numpy as np
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

def clean_train_data(train):
    cols_with_missing = [col for col in train.columns if train[col].isna().sum() > 0 and col != "SurvivalTime"]
    train_clean = train.drop(columns=cols_with_missing)
    train_clean = train_clean.dropna(subset=["SurvivalTime"])
    train_clean = train_clean[train_clean["Censored"] == 0]
    return train_clean

import matplotlib.pyplot as plt
import numpy as np

def plot_y_yhat(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    
    plt.figure(figsize=(6,6))
    plt.scatter(y, yhat, alpha=0.6)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("y vs yhat")
    
    # Draw y=x line
    x_min, x_max = np.min(y), np.max(y)
    plt.plot([x_min, x_max], [x_min, x_max], color='red')
    
    plt.axis('square')
    plt.show()


from sklearn.model_selection import KFold
from sklearn.base import clone
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def baseline_cross_validation(X, y):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),   
            ('regressor', LinearRegression())
    ])
    
    kf = KFold(n_splits=5, shuffle=True)
    models = []

    for train_idx, test_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        model = clone(pipeline)
        model.fit(X_tr, y_tr)
        models.append(model)
    
    return models


import numpy as np

def predict_average(models, X_test):
    
    y_pred = np.zeros(len(X_test))
    for model in models:
        y_pred += model.predict(X_test)
    y_pred /= len(models)
    return y_pred