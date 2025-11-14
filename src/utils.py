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
