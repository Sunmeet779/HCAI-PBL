# project1/forms.py

from django import forms


class UploadDatasetForm(forms.Form):
    file = forms.FileField(
        label="Upload CSV File",
        help_text="Upload a CSV file for model training."
    )


class ModelTrainingForm(forms.Form):
    MODEL_CHOICES = [
        ('logreg', 'Logistic Regression'),
        ('rf', 'Random Forest'),
        ('svm', 'Support Vector Machine'),
    ]

    METRIC_CHOICES = [
        ('accuracy', 'Accuracy'),
        ('f1', 'F1 Score'),
    ]

    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="Select Model"
    )

    label_column = forms.ChoiceField(
        label="Target Column",
        choices=[]  # Dynamically set in the view
    )

    test_size = forms.FloatField(
        label="Test Size",
        min_value=0.1,
        max_value=0.9,
        initial=0.2,
        help_text="Fraction of data to be used for testing (e.g., 0.2 for 20%)."
    )

    c_values = forms.CharField(
        label="C values (comma-separated)",
        required=False,
        help_text="For Logistic Regression or SVM (e.g., 0.01,0.1,1,10)"
    )

    n_estimators_values = forms.CharField(
        label="n_estimators (comma-separated)",
        required=False,
        help_text="For Random Forest (e.g., 10,50,100)"
    )

    max_depth_values = forms.CharField(
        label="max_depth (comma-separated)",
        required=False,
        help_text="For Random Forest (e.g., 3,5,10)"
    )

    metric = forms.ChoiceField(
        choices=METRIC_CHOICES,
        label="Evaluation Metric"
    )
