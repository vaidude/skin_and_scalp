from django.db import models

# Create your models here.
class Register(models.Model):
    name=models.CharField(max_length=50,null=True,blank=True)
    email=models.EmailField(unique=True,null=True,blank=True)
    phone=models.IntegerField(null=True,blank=True)
    gender_choices=(
        ('MALE','male'),
        ('FEMALE','female'),
        ('OTHERS','others'),
    )
    gender=models.CharField(choices=gender_choices,max_length=10)
    age=models.IntegerField(null=True,blank=True)
    password=models.CharField(max_length=8,null=True,blank=True)
    image=models.ImageField(upload_to='user/',null=True,blank=True)
class detect(models.Model):
    image=models.ImageField(upload_to='user/',null=True,blank=True)
    
class scalpdetect(models.Model):
    image=models.ImageField(upload_to='user/',null=True,blank=True)

class Product(models.Model):
    product_name=models.CharField(max_length=50)
    usage=models.CharField(max_length=50)
    dosage=models.CharField(max_length=50)
    price=models.IntegerField()
    
    image=models.ImageField(upload_to='products/',null=True,blank=True)
class Doctor(models.Model):
    name = models.CharField(max_length=200)
    email = models.EmailField()
    password = models.CharField(max_length=100)
    location = models.CharField(max_length=50)
    phone = models.IntegerField()
    profile_pic = models.FileField(upload_to='profile', blank=True, null=True)
    
    specialization = models.CharField(max_length=100, blank=True, null=True)
    qualifications = models.TextField(blank=True, null=True)  # Qualifications info
    years_of_experience = models.PositiveIntegerField(default=0)  # Experience in years
    
    consultation_fee = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
        # Choices for availability
    DAYS_OF_WEEK = [
        ('MONDAY', 'Monday'),
        ('TUESDAY', 'Tuesday'),
        ('WEDNESDAY', 'Wednesday'),
        ('THURSDAY', 'Thursday'),
        ('FRIDAY', 'Friday'),
        ('SATURDAY', 'Saturday'),
        ('SUNDAY', 'Sunday'),
    ]

    # List to store available days of the week
    availability = models.CharField(
        max_length=50,
        choices=DAYS_OF_WEEK,
        blank=True,
        null=True,
        help_text="Select the days of the week available."
    )

    def __str__(self):
        return f"{self.name}, {self.specialization}"
class VetAppointment(models.Model):
    user = models.ForeignKey(Register, on_delete=models.CASCADE)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='appointments')  # Add doctor
    appointment_date = models.DateTimeField() 
    reason_for_visit = models.TextField()  
    status = models.CharField(max_length=50, choices=[('Pending', 'Pending'), ('Confirmed', 'Confirmed'), ('Completed', 'Completed')], default='Pending')

    def __str__(self):
        return f"{self.pet_name} - {self.appointment_date} with {self.doctor.name}"

class scalpproduct(models.Model):
    product_name=models.CharField(max_length=50)
    usage=models.CharField(max_length=50)
    dosage=models.CharField(max_length=50)
    price=models.IntegerField()
    
    image=models.ImageField(upload_to='products/',null=True,blank=True)
    
class Feedback(models.Model):
   RATING_CHOICES = [
      (1, '1'),
      (2, '2'),
      (3, '3'),
      (4, '4'),
      (5, '5'),
      ]
   feedback_text = models.TextField() 
   rating = models.IntegerField(choices=RATING_CHOICES) 
   created_at = models.DateTimeField(auto_now_add=True)
   email = models.EmailField()
   def str(self):
      return f"Rating: {self.rating}, feedback: {self.feedback_text[:50]}..."
  
class Room(models.Model):
    room_name = models.CharField(max_length=50)

    def __str__(self):
        return self.room_name


class Message(models.Model):
    room = models.ForeignKey(Room, on_delete=models.CASCADE)
    sender = models.CharField(max_length=50)
    message = models.TextField()

    def __str__(self):
        return f"{str(self.room)} - {self.sender}"