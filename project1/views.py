import os
import joblib
import pandas as pd
import numpy as np
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadDatasetForm, ModelTrainingForm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


def parse_hyperparams(param_str, param_type=float):
    if not param_str:
        return []
    try:
        return [param_type(x.strip()) if x.strip().lower() != 'none' else None for x in param_str.split(",")]
    except Exception as e:
        print(f"Hyperparam parsing error: {e}")
        return []


def index(request):
    dataset_preview = None
    result = None
    columns = request.session.get('columns', [])
    errors = []
    model_file_url = None

    if request.method == 'POST':
        upload_form = UploadDatasetForm(request.POST, request.FILES)
        train_form = ModelTrainingForm(request.POST)
        train_form.fields['label_column'].choices = [(col, col) for col in columns]

        if 'file' in request.FILES and upload_form.is_valid():
            uploaded_instance = upload_form.save()
            file_path = uploaded_instance.file.path

            try:
                df = pd.read_csv(file_path)
                dataset_preview = df.head().to_html(classes='table table-bordered table-sm')
                columns = df.columns.tolist()
                request.session['dataset_path'] = file_path
                request.session['columns'] = columns
                train_form.fields['label_column'].choices = [(col, col) for col in columns]
            except Exception as e:
                errors.append(f"Failed to read CSV: {str(e)}")

        elif train_form.is_valid():
            file_path = request.session.get('dataset_path')
            if not file_path or not os.path.exists(file_path):
                errors.append("No dataset uploaded or file missing.")
            else:
                try:
                    df = pd.read_csv(file_path)
                    label = train_form.cleaned_data['label_column']

                    if label not in df.columns:
                        errors.append("Selected label column not in dataset.")
                    else:
                        X = df.drop(columns=[label])
                        y = df[label]

                        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

                        numeric_transformer = SimpleImputer(strategy='mean')
                        categorical_transformer = Pipeline([
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))
                        ])

                        preprocessor = ColumnTransformer(transformers=[
                            ('num', numeric_transformer, numeric_cols),
                            ('cat', categorical_transformer, categorical_cols)
                        ])

                        stratify_param = y if y.nunique() > 1 and min(y.value_counts()) > 1 else None

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=train_form.cleaned_data['test_size'],
                            random_state=42, stratify=stratify_param
                        )

                        model_choice = train_form.cleaned_data['model']
                        metric = train_form.cleaned_data['metric']

                        c_values = parse_hyperparams(train_form.cleaned_data.get('c_values'), float)
                        n_estimators_values = parse_hyperparams(train_form.cleaned_data.get('n_estimators_values'), int)
                        max_depth_values = parse_hyperparams(
                            train_form.cleaned_data.get('max_depth_values'),
                            lambda x: int(x) if x and x.lower() != 'none' else None
                        )

                        if model_choice in ['logreg', 'svm'] and not c_values:
                            c_values = [1.0]
                        if model_choice == 'rf':
                            if not n_estimators_values:
                                n_estimators_values = [100]
                            if not max_depth_values:
                                max_depth_values = [None]

                        best_score = -np.inf
                        best_params = {}
                        best_model = None

                        if model_choice == 'logreg':
                            for c in c_values:
                                model = LogisticRegression(C=c, max_iter=1000, random_state=42)
                                pipeline = Pipeline([
                                    ('preprocessor', preprocessor),
                                    ('classifier', model)
                                ])
                                pipeline.fit(X_train, y_train)
                                preds = pipeline.predict(X_test)
                                score = accuracy_score(y_test, preds) if metric == 'accuracy' else f1_score(y_test, preds, average='weighted')

                                if score > best_score:
                                    best_score = score
                                    best_model = pipeline
                                    best_params = {'C': c}

                        elif model_choice == 'svm':
                            for c in c_values:
                                model = SVC(C=c, probability=True, random_state=42)
                                pipeline = Pipeline([
                                    ('preprocessor', preprocessor),
                                    ('classifier', model)
                                ])
                                pipeline.fit(X_train, y_train)
                                preds = pipeline.predict(X_test)
                                score = accuracy_score(y_test, preds) if metric == 'accuracy' else f1_score(y_test, preds, average='weighted')

                                if score > best_score:
                                    best_score = score
                                    best_model = pipeline
                                    best_params = {'C': c}

                        elif model_choice == 'rf':
                            for n_est in n_estimators_values:
                                for max_d in max_depth_values:
                                    model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
                                    pipeline = Pipeline([
                                        ('preprocessor', preprocessor),
                                        ('classifier', model)
                                    ])
                                    pipeline.fit(X_train, y_train)
                                    preds = pipeline.predict(X_test)
                                    score = accuracy_score(y_test, preds) if metric == 'accuracy' else f1_score(y_test, preds, average='weighted')

                                    if score > best_score:
                                        best_score = score
                                        best_model = pipeline
                                        best_params = {'n_estimators': n_est, 'max_depth': max_d}

                        else:
                            errors.append("Invalid model choice.")

                        if best_model:
                            param_str = ", ".join(f"{k}={v}" for k, v in best_params.items() if v is not None)
                            result = f"Best {metric.title()} Score: {best_score:.4f} with parameters: {param_str}"

                            # Save model to disk
                            model_path = os.path.join(settings.MEDIA_ROOT, "best_model.pkl")
                            joblib.dump(best_model, model_path)
                            model_file_url = os.path.join(settings.MEDIA_URL, "best_model.pkl")

                except Exception as e:
                    errors.append(f"Training error: {str(e)}")
        else:
            errors.append("Form invalid. Please check inputs.")
    else:
        upload_form = UploadDatasetForm()
        train_form = ModelTrainingForm()
        train_form.fields['label_column'].choices = [(col, col) for col in columns]

    context = {
        'upload_form': upload_form,
        'train_form': train_form,
        'dataset_preview': dataset_preview,
        'result': result,
        'errors': errors,
        'columns': columns,
        'model_file_url': model_file_url,
    }

    return render(request, 'upload_train.html', context)
