"""
Definition of views.
"""

from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest
from django.views.decorators.csrf import csrf_protect
from django.core.files.storage import default_storage,FileSystemStorage

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )

@csrf_protect
def upload_file(request):
   if request.method == 'POST':
      myfile = request.FILES['file'];

      #Use Default Storage
      #file_name = default_storage.save(myfile.name, myfile);

      #Use FileSystem Storage
      folder='app/bin/uploadfolder/' 
      fs = FileSystemStorage(location=folder) #defaults to MEDIA_ROOT  
      file_name = fs.save(myfile.name, myfile)

      print('file uploaded successfully');
      print(myfile.name);
      return render(request,'app/result.html',{ 'title':'Test Result', 'year':datetime.now().year, 'mldata':myfile.name });

def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'Your application description page.',
            'year':datetime.now().year,
        }
    )


"""
#  Saving POST'ed file to storage
file = request.FILES['myfile']
file_name = default_storage.save(file.name, file)

#  Reading file from storage
file = default_storage.open(file_name)
file_url = default_storage.url(file_name)
"""