import os
from django.conf import settings
from django.shortcuts import render
from django.contrib import messages

from Users.models import UserRegisteredTable

# Create your views here.

def userRegister(request):
    if request.method == 'POST':
        # Extract data from the request
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        loginid = request.POST.get('loginid')
        mobile = request.POST.get('mobile')
        locality = request.POST.get('locality')  # Locality
        state = request.POST.get('state')  # State

        user = UserRegisteredTable(
            name=name,
            email=email,
            password=password,  # Password will be hashed in the model's save method
            loginid=loginid,
            mobile=mobile,
            locality=locality,
            state=state,
        )
        try:
            if user.full_clean:
                user.save()

                messages.success(request, 'Registration successful!.')
                return render(request,'register.html')  # Redirect to the login page or another page as needed
            else:
                messages.error(request,'Entered data is inavalid')
                return render(request,'register.html')
        except:
            messages.error(request,'Entered data is inavalid')
            return render(request,'register.html')


    return render(request, 'register.html')

def userLoginCheck(request):
    if request.method=="POST":
        loginid=request.POST['loginid']
        password=request.POST['password']
        print(loginid,password)
        try:
            user=UserRegisteredTable.objects.get(loginid=loginid,password=password)
            status=user.status
            print(status)
            if status=='activated':
                return render(request,'users/userHome.html')
            else:
                messages.error(request,'Status Not Activated')
                return render(request,'userLogin.html')
        except:
            messages.error(request,'Invalid details')
            return render(request,'userLogin.html')
    else:
        return render(request,'userLogin.html')
    
from django.core.files.storage import FileSystemStorage    

def prediction(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Save the uploaded file to a temporary location
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)
        
        # Construct the full file path for prediction
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        print(f"Uploaded file path: {file_path}")  # Debugging statement

        # Call your model's prediction function here, passing the file path
        predicted_label = predictions(file_path)  # Replace with your actual function
        return render(request,'users/predictionForm.html',{'Prediction':predicted_label})

    return render(request,'users/predictionForm.html')


from Users.utility.requirement  import main, predictions 
def classificationView(request):
    accuracy=main()
    return render(request,'users/classificationView.html',context={'accuracy':accuracy})


            

