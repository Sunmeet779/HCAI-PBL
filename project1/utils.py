import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


def preprocess_dataframe(df, target_column):
    df = df.copy()
    df = df.dropna()

    feature_cols = [col for col in df.columns if col != target_column]

    # Encode categorical features
    for col in feature_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Encode target if categorical
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column].astype(str))

    return df, feature_cols


def train_and_evaluate(df, feature_cols, target_col, model_name, test_size, metric,
                       c_values=None, n_estimators_values=None, max_depth_values=None):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    best_score = -1
    best_params = {}
    best_model = None

    metric_func = accuracy_score if metric == 'accuracy' else f1_score

    if model_name == 'logreg':
        c_list = c_values if c_values else [1.0]
        for c in c_list:
            model = LogisticRegression(C=c, max_iter=200)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = metric_func(y_test, preds)
            if score > best_score:
                best_score = score
                best_params = {'C': c}
                best_model = model

    elif model_name == 'rf':
        n_list = n_estimators_values if n_estimators_values else [100]
        depth_list = max_depth_values if max_depth_values else [None]
        for n in n_list:
            for d in depth_list:
                model = RandomForestClassifier(n_estimators=n, max_depth=d)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = metric_func(y_test, preds)
                if score > best_score:
                    best_score = score
                    best_params = {'n_estimators': n, 'max_depth': d}
                    best_model = model

    elif model_name == 'svm':
        c_list = c_values if c_values else [1.0]
        for c in c_list:
            model = SVC(C=c)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = metric_func(y_test, preds)
            if score > best_score:
                best_score = score
                best_params = {'C': c}
                best_model = model
    else:
        raise ValueError("Unsupported model selected")

    return {
        'best_score': best_score,
        'best_params': best_params,
        'model': best_model
    }


def generate_visualizations(df, target_column):
    plt.switch_backend('Agg')

    imgs_html = []

    # Target distribution plot
    fig1, ax1 = plt.subplots()
    sns.countplot(x=target_column, data=df, ax=ax1)
    ax1.set_title('Target Column Distribution')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    img1 = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)
    imgs_html.append(f'<img src="data:image/png;base64,{img1}" alt="Target Distribution" />')

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] > 1:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Feature Correlation Heatmap')
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        img2 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)
        imgs_html.append(f'<img src="data:image/png;base64,{img2}" alt="Correlation Heatmap" />')

    return '<br>'.join(imgs_html)
