from django.shortcuts import render
from .models import Patient
from .ml_model import predict_cancer


def home(request):

    result = None
    confidence = None
    patient = None

    if request.method == 'POST':

        name = request.POST['name']
        age = request.POST['age']
        gender = request.POST['gender']
        image = request.FILES['image']

        # Prediction
        result, confidence = predict_cancer(image)

        # Save to SQLite database
        patient = Patient.objects.create(
            name=name,
            age=age,
            gender=gender,
            image=image,
            prediction=result,
            confidence=confidence
        )

    # Get all previous records
    patients = Patient.objects.all().order_by('-date')

    return render(request, 'main/index.html', {
        'result': result,
        'confidence': confidence,
        'patient': patient,
        'patients': patients
    })