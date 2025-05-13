import pandas as pd
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

def index(request):
    return render(request, 'project1/index.html')

def upload_file(request):
    if request.method == 'POST' and request.FILES['dataset']:
        dataset = request.FILES['dataset']
        fs = FileSystemStorage()
        filename = fs.save(dataset.name, dataset)
        df = pd.read_csv(fs.path(filename))

        request.session['data_path'] = fs.path(filename)  # Save path for reuse
        return render(request, 'project1/display.html', {'df': df.to_html()})
    return render(request, 'project1/upload.html')
