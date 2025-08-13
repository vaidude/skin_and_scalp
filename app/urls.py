from django.urls import path
from .import views
urlpatterns=[
    path('',views.index,name='index'),
    path('home/',views.home,name='home'),
    path('login/',views.login,name='login'),
    path('profile/',views.profile,name='profile'),
    path('register/',views.register,name='register'),
    path('editprofile/',views.editprofile,name='editprofile'),
    path('adlogin/',views.adlogin,name='adlogin'),
    path('adhome/',views.adhome,name='adhome'),
    path('userslist/',views.userslist,name='userslist'),
    path('logout/',views.logout,name='logout'),
    path('deleteuser/<int:id>/',views.deleteuser,name='deleteuser'),
    path('adeditprofile/<int:uid>/',views.adeditprofile,name='adeditprofile'),
    path('detect/',views.detectscalp,name='detect'),
    path('scalpdetect/',views.detectscalp,name='scalpdetect'),
    path('addproduct/',views.addproduct,name='addproduct'),
    path('addscalpproduct/',views.addscalpproduct,name='addscalpproduct'),
    path('skinproductlist/',views.skinproductlist,name='skinproductlist'),
    path('scalpproductlist/',views.scalpproductlist,name='scalpproductlist'),
    path('userskinproductlist/',views.userskinproductlist,name='userskinproductlist'),
    path('userscalpproductlist/',views.userscalpproductlist,name='userscalpproductlist'),
    path('deleteskinproduct/<int:skin_id>/',views.deleteskinproduct,name='deleteskinproduct'),
    path('deletescalpproduct/<int:scalp_id>/',views.deletescalpproduct,name='deletescalpproduct'),
    path('edit_product/<int:id>/',views.edit_product,name='edit_product'),
    path('edit_scalpproduct/<int:id>/',views.edit_scalpproduct,name='edit_scalpproduct'),
    
    path('doctorregister/', views.register_doctor, name='doctorregister'),
    path('doctorlogin/', views.doctorlogin, name='doctorlogin'),
    path('doctorhome/', views.doctorhome, name='doctorhome'),
    path('doctorprofile/', views.doctorprofile, name='doctorprofile'),
    path('editdoctorprofile/', views.editdoctorprofile, name='editdoctorprofile'),
    
    path('book-appointment/', views.book_appointment, name='book_appointment'),
    path('appointment-success/', views.appointment_success, name='appointment_success'),
    path('appointments/', views.doctor_appointments, name='doctor_appointments'),
    
    path('skin/', views.upload_image, name='skin'),
    path('scalpanalysis/', views.scalp_analysis_view, name='scalpanalysis'),
    
    
    path('feedback/',views.feedback,name='feedback'),
    path('feedbacklist/',views.feedback_list,name='feedbacklist'),
    
    path("chat/", views.HomeView, name="chat"),
    path("<str:room_name>/<str:username>/", views.RoomView, name="room"),
]
