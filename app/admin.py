from django.contrib import admin
from .import models
# Register your models here.
admin.site.register(models.Register)
admin.site.register(models.Product)
admin.site.register(models.scalpproduct)
admin.site.register(models.detect)
admin.site.register(models.scalpdetect)
admin.site.register(models.Feedback)
admin.site.register(models.Room)