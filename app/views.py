from http.client import HTTPResponse
from django.shortcuts import render,redirect,Http404,get_object_or_404
from .models import Register,detect,Product,scalpproduct
from .models import*
# Create your views here.
def index(request):
    return render(request,'index.html')

def home(request):
     return render(request,'home.html')
def register(request):
    if request.method=='POST':
        name=request.POST.get('username')
        email=request.POST.get('email')
        phone=request.POST.get('phone')
        gender=request.POST.get('gender')
        age=request.POST.get('age')
        password=request.POST.get('password')
        img=request.FILES.get('img')
        Register(name=name,email=email,phone=phone,gender=gender,age=age,password=password,image=img).save()
        return redirect('index')
    return render(request,'register.html')
def login(request):
        if request.method=='POST':
            email=request.POST.get('email')
            password=request.POST.get('password')
            user=Register.objects.filter(email=email,password=password).first()
            if user:
                request.session['email']=email
                return redirect('home')
            else:
                return render(request,'login.html',{'error':'invalid credentials'}) 
        return render(request,'login.html')  
def  profile(request):
    email=request.session.get('email')
    if email:
         user=Register.objects.get(email=email) 
         return render(request,'profile.html',{'user':user})
    else:
        return redirect('login')      
def  editprofile(request):
    if request.method=='POST':
        email=request.session.get('email')
        user=Register.objects.get(email=email)
        user.name=request.POST.get('name',user.name)
        user.password=request.POST.get('password',user.password)
        user.phone=request.POST.get('phone',user.phone)
        user.gender=request.POST.get('gender',user.gender)
        user.age=request.POST.get('age',user.age)
        user.image=request.FILES.get('image',user.image)
        user.save()
        return redirect('home')
    return render(request,'profile.html')

def adlogin(request):
    if request.method=='POST':
        name=request.POST.get('name')
        password=request.POST.get('password')
        u='admin'
        p='admin'
        if name==u:
            if password==p:
                return redirect('adhome')
            else:
                 return render(request,'adlogin.html')
        
        return render(request,'adlogin.html')
    return render(request,'adlogin.html')
def adhome(request):
    user=Register.objects.all().count()
    return render(request,'adhome.html',{'user':user})


def userslist(request):
    user=Register.objects.all()
    return render (request,'userslist.html',{'user':user})
def deleteuser(request,id):
    try:
        user=Register.objects.get(id=id)
        user.delete()
        return redirect('userslist')
    except Register.DoesNotExist:
            raise Http404("user not found")
def  adeditprofile(request,uid):
    if request.method=='POST':
        user=Register.objects.get(id=uid)
        user.email=request.POST.get('email',user.email)
        user.name=request.POST.get('name',user.name)
        user.password=request.POST.get('password',user.password)
        user.phone=request.POST.get('phone',user.phone)
        user.gender=request.POST.get('gender',user.gender)
        user.age=request.POST.get('age',user.age)
        user.image=request.FILES.get('image',user.image)
        user.save()
        return redirect('adhome')
    else:
        user=Register.objects.get(id=uid)
        return render(request,'adeditprofile.html',{'user':user})

def logout(request):
    request.session.flush()
    return redirect('index')
def detectscalp(request):
    if request.method == 'POST':
        img= request.FILES.get('img')
        im=detect(image=img)
        im.save()
        return redirect('home')
    return render(request,'detect.html')


def addproduct(request):
    if request.method=='POST':
       product_name=request.POST.get('product_name')
       usage=request.POST.get('usage')
       dosage=request.POST.get('dosage')
       
       price=request.POST.get('price')
       image=request.FILES.get('image')
       Product(product_name=product_name,
                usage=usage,
                dosage=dosage,
                price=price,
                
                image=image,
                ).save()
       return redirect('adhome')
    return render(request,'addproduct.html')

def addscalpproduct(request):
    if request.method=='POST':
       product_name=request.POST.get('product_name')
       usage=request.POST.get('usage')
       dosage=request.POST.get('dosage')
       
       price=request.POST.get('price')
       image=request.FILES.get('image')
       scalpproduct(product_name=product_name,
                usage=usage,
                dosage=dosage,
                price=price,
                
                image=image,
                ).save()
       return redirect('adhome')
    return render(request,'addscalpproduct.html')
def skinproductlist(request):
    skin=Product.objects.all()
    return render(request,'skinproductlist.html',{'skin':skin})
def scalpproductlist(request):
    scalp=scalpproduct.objects.all()
    return render(request,'scalpproductlist.html',{'scalp':scalp})
def userskinproductlist(request):
    userskin=Product.objects.all()
    return render(request,'userskinproductlist.html',{'userskin':userskin})
def userscalpproductlist(request):
    userscalp=scalpproduct.objects.all()
    return render(request,'userscalpproductlist.html',{'userscalp':userscalp})
def deleteskinproduct(request,skin_id):
    skin=Product.objects.get(id=skin_id)
    skin.delete()
    return redirect('skinproductlist')
def deletescalpproduct(request,scalp_id):
    scalp=scalpproduct.objects.get(id=scalp_id)
    scalp.delete()
    return redirect('scalpproductlist')
def edit_product(request, id):
    product = get_object_or_404(Product, id=id)
    
    if request.method == 'POST':
        product.product_name = request.POST.get('product_name')
        product.usage = request.POST.get('usage')
        product.dosage = request.POST.get('dosage')
        
        product.price = request.POST.get('price')

        if 'image' in request.FILES:
            product.image = request.FILES['image']
        
        product.save()
        return redirect('skinproductlist')  
    else:
        return render(request, 'edit_product.html', {'product': product})
        
def edit_scalpproduct(request, id):
    scalpproduct = get_object_or_404(Product, id=id)
    
    if request.method == 'POST':
        scalpproduct.product_name = request.POST.get('product_name')
        scalpproduct.usage = request.POST.get('usage')
        scalpproduct.dosage = request.POST.get('dosage')
        
        scalpproduct.price = request.POST.get('price')

        if 'image' in request.FILES:
            scalpproduct.image = request.FILES['image']
        
        scalpproduct.save()
        return redirect('scalpproductlist')  
    else:
        return render(request, 'edit_scalpproduct.html', {'scalpproduct': scalpproduct})
    
from django.contrib import messages
        
def feedback(request):
    if request.method == "POST":
        email=request.POST.get('email')
        feedback_text = request.POST.get('feedback_text')
        rating = request.POST.get('rating')
        if not feedback_text or not rating:
            messages.error(request, "Please fill in all required fields.")
        try:
            rating = int(rating)
            if rating not in [1, 2, 3, 4, 5]:
                raise ValueError("Invalid rating value")
        except (ValueError, TypeError):
            # Handle invalid rating
            messages.error(request, "Invalid rating value. Please select a valid rating.")
            return HTTPResponse(messages)
        feedback = Feedback(
            email=email,
            feedback_text=feedback_text,
            rating=rating
        )
        feedback.save()
        messages.success(request, "Feedback submitted successfully!")
        return redirect('feedback')
    else:
        email=request.session.get('email')
        user=Register.objects.get(email=email) 
        return render(request, 'feedback.html',{'user':user})


def feedback_list(request):
    feed = Feedback.objects.all()
    return render(request, 'feedback_list.html', {'feed': feed})

def HomeView(request):
    if request.method == "POST":
        username = request.POST["username"]
        room = request.POST["room"]
        try:
            existing_room = Room.objects.get(room_name__icontains=room)
        except Room.DoesNotExist:
            r = Room.objects.create(room_name=room)
        return redirect("room", room_name=room, username=username)
    email=request.session.get('email')
    user=Register.objects.get(email=email)
    return render(request, "chat.html",{'user':user})


def RoomView(request, room_name, username):
    try:
        existing_room = Room.objects.get(room_name__icontains=room_name)
    except Room.DoesNotExist:
        return redirect("login")
    get_messages = Message.objects.filter(room=existing_room)
    context = {
        "messages": get_messages,
        "user": username,
        "room_name": existing_room.room_name,
    }

    return render(request, "room.html", context)

def register_doctor(request):
    if request.method == 'POST':
        # Get the form data from the request
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        location = request.POST.get('location')
        phone = request.POST.get('phone')
        specialization = request.POST.get('specialization')
        qualifications = request.POST.get('qualifications')
        years_of_experience = request.POST.get('years_of_experience')
        availability = request.POST.get('availability')
        consultation_fee = request.POST.get('consultation_fee')
        
        # Handle profile picture
        profile_pic = request.FILES.get('profile_pic')

        # Create the new doctor record
        doctor = Doctor(
            name=name,
            email=email,
            password=password,
            location=location,
            phone=phone,
            specialization=specialization,
            qualifications=qualifications,
            years_of_experience=years_of_experience,
            availability=availability,
            consultation_fee=consultation_fee,
            profile_pic=profile_pic
        )
        
        # Save the doctor instance to the database
        doctor.save()
        
        # Redirect to a success page or show a confirmation
        return redirect('doctorlogin')  # Example redirect (you can create a doctor list page)
    
    return render(request, 'doctorregister.html')

def doctorlogin(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            user = Doctor.objects.get(email=email, password=password)
        
            request.session['email'] = user.email
           
            return redirect('doctorhome')  # Redirect to a home page or dashboard
        except Doctor.DoesNotExist:
            return render(request, 'doctorlogin.html', {'error': 'Invalid email or password.'})

    
    return render(request, 'doctorlogin.html')

def doctorhome(request):
    return render(request, 'doctorhome.html')
def doctorprofile(request):
    email = request.session.get('email')
    
    if email is not None:
        try:
            user = Doctor.objects.get(email=email)
            return render(request, 'doctorprofile.html', {'user': user})
        except Doctor.DoesNotExist:
            messages.error(request, "User not found.")
            return redirect('doctorlogin')  
    else:
        messages.warning(request, "You need to log in to access your profile.")
        return redirect('doctorlogin') 
def editdoctorprofile(request):
    email = request.session.get('email') 

    if email is not None:
        try:
            user = Doctor.objects.get(email=email)

            if request.method == 'POST':
                # Update the doctor's details
                user.location = request.POST.get('location')
                user.phone = request.POST.get('phone')
                user.specialization = request.POST.get('specialization')
                user.qualifications = request.POST.get('qualifications')
                user.years_of_experience = request.POST.get('years_of_experience')
                user.consultation_fee = request.POST.get('consultation_fee')
                selected_days = request.POST.getlist('availability')
                user.availability = ', '.join(selected_days)

                user.save()
                messages.success(request, "Profile updated successfully!")
                return redirect('doctorprofile')
            
            return render(request, 'doctorprofile.html', {'user': user})

        except Doctor.DoesNotExist:
            messages.error(request, "User not found.")
            return redirect('doctorlogin')
    else:
        messages.warning(request, "You need to log in to access your profile.")
        return redirect('doctorlogin')

def book_appointment(request):
    doctors = Doctor.objects.all()
    email = request.session.get('email')
    
    if email:
        # Retrieve the user based on the email
        user = get_object_or_404(Register, email=email)# Fetch all doctors

    if request.method == "POST":

        doctor_id = request.POST.get("doctor")  # Get selected doctor ID
        doctor = Doctor.objects.get(id=doctor_id)
        appointment_date = request.POST.get("appointment_date")
        reason_for_visit = request.POST.get("reason_for_visit")

        # Create a new appointment
        appointment = VetAppointment(
            
            user=user,
            doctor=doctor,
            appointment_date=appointment_date,
            reason_for_visit=reason_for_visit,
        )
        appointment.save()

        # Redirect to a success page
        return redirect('appointment_success')

    # If it's a GET request, show the available doctors
    return render(request, 'book_appointment.html', {'doctors': doctors})


def appointment_success(request):
    email = request.session.get('email')
    
    if email:
        # Retrieve the user based on the email
        user = get_object_or_404(Register, email=email)
        appointment = VetAppointment.objects.filter(user=user)
        return render(request, 'appointments.html', {'appointment': appointment})
    return render(request, 'appointments.html')



def doctor_appointments(request):
  
    doctor_email = request.session.get('email')  # Assume the email is stored in the session
    if not doctor_email:
        return redirect('doctorlogin') 
    
    doctor=Doctor.objects.get(email=doctor_email)
    appointments = VetAppointment.objects.filter(doctor=doctor).order_by('appointment_date')
    
    # Handle POST request to update appointment status
    if request.method == 'POST':
        appointment_id = request.POST.get('appointment_id')
        new_status = request.POST.get('status')
        appointment = get_object_or_404(VetAppointment, id=appointment_id, doctor__email=doctor_email)
        if new_status in dict(VetAppointment._meta.get_field('status').choices):
            appointment.status = new_status
            appointment.save()
        return redirect('doctor_appointments')
    
    return render(request, 'doctorappoinments.html', {'appointments': appointments})

#skin analysis
import os
import cv2
import numpy as np
from django.conf import settings
from django.shortcuts import render
from deepface import DeepFace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import openai

# Path to the model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model.h5')
model = load_model(MODEL_PATH, custom_objects={
    'top_3_accuracy': TopKCategoricalAccuracy(k=3),
    'top_2_accuracy': TopKCategoricalAccuracy(k=2)
})

openai.api_key = "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"  

# Upload and analyze image
def upload_image(request):
    if request.method == "POST" and request.FILES['image']:
        uploaded_image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)

        # Save the image
        with open(image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Analyze the uploaded image
        context = analyze_image(image_path)
        return render(request, "skinresult.html", context)

    return render(request, "upload.html")


def analyze_image(image_path):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    print(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict skin condition
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    skin_conditions = ["Healthy", "Acne", "Eczema", "Psoriasis", "Rosacea", "Melanoma"]
    predicted_condition = skin_conditions[predicted_class[0]]

    # Perform race analysis using DeepFace
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['race'], enforce_detection=False)
        dominant_race = analysis[0]['dominant_race']
    except Exception as e:
        dominant_race = "Unknown"

    # Generate skincare recommendations
    recommendations = generate_recommendations(dominant_race, predicted_condition)

    return {
        "image_path": image_path,
        "predicted_condition": predicted_condition,
        "dominant_race": dominant_race,
        "recommendations": recommendations,
    }


def generate_recommendations(skin_type, skin_condition):
    prompt = f"""
    Provide detailed skincare product recommendations for a person with the following attributes:
    - Skin Type: {skin_type}
    - Skin Condition: {skin_condition}

    Include:
    1. A list of skincare products (e.g., cleansers, moisturizers, serums, sunscreens).
    2. Dosages or application instructions for each product.
    3. General skincare advice tailored to the skin type and condition.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skincare expert providing personalized recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating recommendations: {e}"


#scalp analysis
import cv2
import numpy as np
import openai
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings

# Set OpenAI API Key
openai.api_key = "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" 

def scalp_analysis_view(request):
    if request.method == "POST" and request.FILES["image"]:
        # Step 1: Handle uploaded image
        uploaded_file = request.FILES["image"]
        file_path = default_storage.save(f"uploads/{uploaded_file.name}", uploaded_file)
        file_full_path = f"{settings.MEDIA_ROOT}/{file_path}"

        # Step 2: Read the uploaded image
        img = cv2.imread(file_full_path)
        status = ""

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3: Detect hair region
        hair_mask = cv2.inRange(gray_img, 0, 100)  # Adjust upper limit for hair detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)

        # Enhance dandruff detection
        blurred_hair = cv2.GaussianBlur(hair_mask, (5, 5), 0)
        dandruff_mask = cv2.adaptiveThreshold(
            blurred_hair, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Find and filter contours
        contours, _ = cv2.findContours(dandruff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 30
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Draw contours on the image
        detection_img = img.copy()
        cv2.drawContours(detection_img, filtered_contours, -1, (0, 255, 0), 2)

        # Determine dandruff severity based on contour count
        if len(filtered_contours) > 30:
            status = "Major Dandruff Scalp"  # More contours = more dandruff
        elif len(filtered_contours) > 10:
            status = "Severe Dandruff Scalp"  # Moderate number of contours = severe dandruff
        else:
            status = "No Dandruff Scalp"  # Few contours = no dandruff

        # Generate recommendations using OpenAI
        recommendations = generate_scalp_recommendations(status)

        # Save the processed image
        processed_file_path = f"uploads/processed_{uploaded_file.name}"
        processed_full_path = f"{settings.MEDIA_ROOT}/{processed_file_path}"
        cv2.imwrite(processed_full_path, detection_img)

        # Return results to the template
        return render(request, "scalpresult.html", {
            "original_image": file_path,
            "processed_image": processed_file_path,
            "dandruff_status": status,
            "recommendations": recommendations,
        })

    return render(request, "scalp_analysis.html")

def generate_scalp_recommendations(dandruff_status):
    prompt = f"""
    Provide detailed scalp care product recommendations for a person with the following condition:
    - Scalp Condition: {dandruff_status}

    Include:
    1. A list of scalp care products (e.g., shampoos, conditioners, treatments).
    2. Dosages or application instructions for each product.
    3. General scalp care advice tailored to the severity of the scalp condition.

    Format the output as:
    Recommendations:
    - Product 1: Name, dosage/instructions
    - Product 2: Name, dosage/instructions
    General Advice:
    - Advice 1
    - Advice 2
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[ 
                {"role": "system", "content": "You are a skincare expert providing personalized scalp care recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        return f"Error generating recommendations: {e}"
