from django.urls import path
from . import views

urlpatterns = [
    path('', views.build_model),
    #path('index/',views.index),
    path('out/',views.out),
    path('visual',views.visual),
] 
