import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns

def run_logistic_with_lambda(lambda_coeff):
    # Load & clean data
    df = load_penguins().dropna()
    X = pd.get_dummies(df.drop(columns=['species']))
    y = df['species']

    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train with L1 penalty to enforce sparsity
    clf = LogisticRegression(penalty='l1', C=1/(lambda_coeff+1e-8), solver='saga', max_iter=5000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Number of non-zero coefficient features
    non_zero = np.sum(np.any(clf.coef_ != 0, axis=0))
    feature_names = X.columns[clf.coef_.any(axis=0)]
    coefs = np.mean(np.abs(clf.coef_[:, clf.coef_.any(axis=0)]), axis=0)

    # Plot top features
    plt.figure(figsize=(8, 4))
    order = np.argsort(coefs)[::-1]
    sns.barplot(x=coefs[order], y=[feature_names[i] for i in order], palette="viridis")
    plt.title("Average absolute coefficients")
    plt.tight_layout()
    os.makedirs("project3/static/project3", exist_ok=True)
    plot_path = "project3/static/project3/logistic_coefs.png"
    plt.savefig(plot_path)
    plt.close()

    return acc, non_zero, 'project3/logistic_coefs.png'
