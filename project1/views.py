import tempfile
import pandas as pd
import numpy as np
from django.shortcuts import render
from .forms import UploadDatasetForm, ModelTrainingForm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def parse_hyperparams(param_str, param_type=float):
    """Parse comma-separated hyperparameter string into a list."""
    if not param_str:
        return []
    try:
        return [param_type(x.strip()) if x.strip().lower() != 'none' else None for x in param_str.split(",")]
    except:
        return []

def index(request):
    dataset_preview = None
    result = None
    columns = request.session.get('columns', [])
    errors = []
    if request.method == 'POST':
        upload_form = UploadDatasetForm(request.POST, request.FILES)
        train_form = ModelTrainingForm(request.POST)
        train_form.fields['label_column'].choices = [(col, col) for col in columns]
        # Handle dataset upload
        if upload_form.is_valid() and 'file' in request.FILES:
            csv_file = request.FILES['file']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                for chunk in csv_file.chunks():
                    tmp.write(chunk)
                file_path = tmp.name
            try:
                df = pd.read_csv(file_path)
                dataset_preview = df.head().to_html(classes='table table-bordered table-sm')
                columns = df.columns.tolist()
                request.session['dataset_path'] = file_path
                request.session['columns'] = columns
                train_form.fields['label_column'].choices = [(col, col) for col in columns]
            except Exception as e:
                errors.append(f"Failed to read CSV: {str(e)}")
        # Handle model training
        elif train_form.is_valid():
            file_path = request.session.get('dataset_path')
            if not file_path:
                errors.append("No dataset uploaded.")
            else:
                try:
                    df = pd.read_csv(file_path)
                    label = train_form.cleaned_data['label_column']
                    if label not in df.columns:
                        errors.append("Selected label column not in dataset.")
                    else:
                        X = df.drop(columns=[label])
                        y = df[label]
                        # Split and preprocess
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
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=train_form.cleaned_data['test_size'], random_state=42
                        )
                        model_choice = train_form.cleaned_data['model']
                        metric = train_form.cleaned_data['metric']
                        c_values = parse_hyperparams(train_form.cleaned_data.get('c_values'), float)
                        n_estimators_values = parse_hyperparams(train_form.cleaned_data.get('n_estimators_values'), int)
                        max_depth_values = parse_hyperparams(train_form.cleaned_data.get('max_depth_values'), lambda x: int(x) if x != 'None' else None)
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
                        for c in c_values if model_choice in ['logreg', 'svm'] else [None]:
                            for n_est in n_estimators_values if model_choice == 'rf' else [None]:
                                for max_d in max_depth_values if model_choice == 'rf' else [None]:
                                    # Build model
                                    if model_choice == 'logreg':
                                        model = LogisticRegression(C=c, max_iter=1000)
                                    elif model_choice == 'svm':
                                        model = SVC(C=c, probability=True)
                                    elif model_choice == 'rf':
                                        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d)
                                    else:
                                        continue
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
                                        best_params = {
                                            'C': c, 'n_estimators': n_est, 'max_depth': max_d
                                        }
                        param_str = ", ".join(f"{k}={v}" for k, v in best_params.items() if v is not None)
                        result = f"Best {metric.title()} Score: {best_score:.4f} with parameters: {param_str}"
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
    }
    return render(request, 'upload_train.html', context)