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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.base import clone
import numpy as np
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.base import clone

def baseline_cv_predict(df, keep_missing=True, n_splits=5, c=0):
    # Optionally drop indicator columns
    if not keep_missing:
        df = df.drop(columns=[col for col in df.columns if "_missing" in col])
    
    # Define features and target
    X = df.drop(columns=["id","SurvivalTime", "Censored"])
    y = df["SurvivalTime"]
    censored = df["Censored"]
    ids = df["id"]
    
    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),   
        ('regressor', LinearRegression())
    ])
    
    # Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    y_pred_total = np.zeros(len(X))
    
    for train_idx, test_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        model = clone(pipeline)
        model.fit(X_tr, y_tr)
        models.append(model)
        
        # Predict on test fold
        X_te = X.iloc[test_idx]
        y_pred_total[test_idx] = model.predict(X_te)
    
    # Compute custom MSE
    mse = error_metric(y.values, y_pred_total, censored.values)
    
    # Build predictions DataFrame
    df_pred = pd.DataFrame({
        "id": ids,
        "SurvivalTime": y,
        "Censored": censored,
        "y_pred": y_pred_total
    })
    
    return models, y_pred_total, mse, df_pred


def select_polynomial_model(
    X_tr, y_tr,
    degrees=[1, 2, 3, 4, 5],
    n_splits=5,
    shuffle=True
):

    cv_scores = {}
    best_score = np.inf
    best_degree = None
    best_model = None

    for d in degrees:
        # Pipeline: standardizzazione -> espansione polinomiale -> regressione lineare
        pipe_poly = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("lin", LinearRegression())
        ])

        kfold = KFold(n_splits=n_splits, shuffle=shuffle)

        # out-of-fold predictions sul training
        y_tr_oof = cross_val_predict(pipe_poly, X_tr, y_tr, cv=kfold)

        # tua metrica (es. MSE)
        cmse_cv = error_metric(y_tr, y_tr_oof,c=0)
        cv_scores[d] = cmse_cv

        # tiene traccia del modello migliore
        if cmse_cv < best_score:
            best_score = cmse_cv
            best_degree = d
            # rifitta sul training completo
            best_model = pipe_poly.fit(X_tr, y_tr)

    return best_model, best_degree, cv_scores

# k-Nearest Neighbors
#best_model_fitted, best_params, cv_scores
def select_knn_model(
    X_tr, y_tr,
    k_list=[3, 5, 7, 9, 11],
    weights_list=["uniform", "distance"],
    n_splits=5,
    shuffle=True
):

    cv_scores = {}
    best_score = np.inf
    best_params = None
    best_model = None

    for k in k_list:
        for w in weights_list:
            pipe_knn = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(
                    n_neighbors=k,
                    weights=w
                ))
            ])

            kfold = KFold(n_splits=n_splits, shuffle=shuffle)

            # out-of-fold predictions sul training
            y_tr_oof = cross_val_predict(pipe_knn, X_tr, y_tr, cv=kfold)

            cmse_cv = error_metric(y_tr, y_tr_oof,c=0)
            cv_scores[(k, w)] = cmse_cv

            if cmse_cv < best_score:
                best_score = cmse_cv
                best_params = {"k": k, "weights": w}
                best_model = pipe_knn.fit(X_tr, y_tr)

    return best_model, best_params, cv_scores


def evaluate_model_cv(model, X, y, n_splits=5, shuffle=True):

    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    fold_errors = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)

        y_hat = m.predict(X_te)
        err = error_metric(y_te.values, y_hat, c=0)
        fold_errors.append(err)

    return fold_errors
