import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import CSVUploadForm, TargetSelectForm, ModelSelectForm
from . import utils


def index(request):
    if request.method == 'POST':

        # Step 1: Upload CSV file
        if 'upload_csv' in request.POST:
            upload_form = CSVUploadForm(request.POST, request.FILES)
            if upload_form.is_valid():
                csv_file = upload_form.cleaned_data['file']
                try:
                    df = pd.read_csv(csv_file)
                except Exception:
                    messages.error(request, "Invalid CSV file or file could not be read.")
                    return redirect('project1:index')

                request.session['csv_data'] = df.to_json()
                columns = df.columns.tolist()

                target_form = TargetSelectForm(column_choices=columns)
                return render(request, 'upload_train.html', {
                    'step': 2,
                    'target_form': target_form
                })
            else:
                messages.error(request, "Please upload a valid CSV file.")
                return redirect('project1:index')

        # Step 2: Select Target column
        elif 'select_target' in request.POST:
            csv_json = request.session.get('csv_data')
            if not csv_json:
                messages.error(request, "Session expired. Please upload the CSV file again.")
                return redirect('project1:index')

            df = pd.read_json(csv_json)
            target_form = TargetSelectForm(column_choices=df.columns.tolist(), data=request.POST)
            if target_form.is_valid():
                target_column = target_form.cleaned_data['target_column']
                request.session['target_column'] = target_column

                model_form = ModelSelectForm()
                return render(request, 'upload_train.html', {
                    'step': 3,
                    'model_form': model_form,
                    'target_column': target_column
                })
            else:
                messages.error(request, "Please select a valid target column.")
                return redirect('project1:index')

        # Step 3: Train Model
        elif 'train_model' in request.POST:
            csv_json = request.session.get('csv_data')
            target_column = request.session.get('target_column')

            if not csv_json or not target_column:
                messages.error(request, "Session expired. Please upload and select target again.")
                return redirect('project1:index')

            df = pd.read_json(csv_json)

            model_form = ModelSelectForm(request.POST)
            if model_form.is_valid():
                model_name = model_form.cleaned_data['model']
                test_size = model_form.cleaned_data['test_size']
                metric = model_form.cleaned_data['metric']

                # Parse hyperparameters
                def parse_floats(s):
                    try:
                        return [float(x.strip()) for x in s.split(',') if x.strip()]
                    except:
                        return None

                c_values = parse_floats(model_form.cleaned_data['c_values']) if model_name in ['logreg', 'svm'] else None
                n_estimators_values = parse_floats(model_form.cleaned_data['n_estimators_values']) if model_name == 'rf' else None
                max_depth_values = parse_floats(model_form.cleaned_data['max_depth_values']) if model_name == 'rf' else None

                try:
                    df_processed, feature_cols = utils.preprocess_dataframe(df, target_column)
                except Exception as e:
                    messages.error(request, f"Error preprocessing data: {e}")
                    return redirect('project1:index')

                try:
                    results = utils.train_and_evaluate(
                        df_processed, feature_cols, target_column,
                        model_name, test_size, metric,
                        c_values=c_values,
                        n_estimators_values=n_estimators_values,
                        max_depth_values=max_depth_values
                    )
                except Exception as e:
                    messages.error(request, f"Error training model: {e}")
                    return redirect('project1:index')

                visualizations = utils.generate_visualizations(df, target_column)

                return render(request, 'upload_train.html', {
                    'step': 4,
                    'results': results,
                    'model_name': dict(model_form.fields['model'].choices).get(model_name, model_name),
                    'metric': dict(model_form.fields['metric'].choices).get(metric, metric),
                    'visualizations': visualizations
                })

            else:
                messages.error(request, "Please correct the errors in the form.")
                target_column = request.session.get('target_column')
                return render(request, 'upload_train.html', {
                    'step': 3,
                    'model_form': model_form,
                    'target_column': target_column
                })

    else:
        upload_form = CSVUploadForm()
        return render(request, 'upload_train.html', {
            'step': 1,
            'upload_form': upload_form
        })
