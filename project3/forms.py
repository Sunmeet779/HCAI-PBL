from django import forms

class LambdaForm(forms.Form):
    lambda_coeff = forms.FloatField(
        min_value=0.0, max_value=1.0, initial=0.1,
        widget=forms.NumberInput(attrs={'step':0.01})
    )
