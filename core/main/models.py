from django.db import models

class Patient(models.Model):

    name = models.CharField(max_length=100)

    age = models.IntegerField()

    gender = models.CharField(max_length=10)

    image = models.ImageField(upload_to='uploads/')

    prediction = models.CharField(max_length=50)

    confidence = models.FloatField()

    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name