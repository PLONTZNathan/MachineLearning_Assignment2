{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "31c158ed-fe41-4801-ab24-ead8e91f4633",
   "metadata": {},
   "outputs": [],
   "source": [
    "def error_metric(y, y_hat, c):\n",
    "    import numpy as np\n",
    "    err = y-y_hat\n",
    "    err = (1-c)*err**2 + c*np.maximum(0,err)**2\n",
    "    return np.sum(err)/err.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "549e0ace-4bf8-4e78-a7ee-e5ad37d352bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_train_data(train):\n",
    "    cols_with_missing = [col for col in train.columns if train[col].isna().sum() > 0 and col != \"SurvivalTime\"]\n",
    "    train_clean = train.drop(columns=cols_with_missing)\n",
    "    train_clean = train_clean.dropna(subset=[\"SurvivalTime\"])\n",
    "    train_clean = train_clean[train_clean[\"Censored\"] == 0]\n",
    "    return train_clean\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5e0bf08e-398d-4d2a-9aaf-80ed02a0ee6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "def plot_y_yhat(y, yhat):\n",
    "    y = np.array(y)\n",
    "    yhat = np.array(yhat)\n",
    "    \n",
    "    plt.figure(figsize=(6,6))\n",
    "    plt.scatter(y, yhat, alpha=0.6)\n",
    "    plt.xlabel(\"True y\")\n",
    "    plt.ylabel(\"Predicted y\")\n",
    "    plt.title(\"y vs yhat\")\n",
    "    \n",
    "    # Draw y=x line\n",
    "    x_min, x_max = np.min(y), np.max(y)\n",
    "    plt.plot([x_min, x_max], [x_min, x_max], color='red')\n",
    "    \n",
    "    plt.axis('square')\n",
    "    plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
