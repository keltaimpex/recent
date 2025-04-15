from django.shortcuts import render

# Create your views here.
import pickle
import numpy as np 

import os
model_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
model = pickle.load(open(model_path, 'rb'))
def home(request):
    price=None
    if request.method == 'POST':
        sqft_living = float(request.POST['sqft_living'])
        bedrooms = int(request.POST['bedrooms'])
        grade = int(request.POST['grade'])
        floors = float(request.POST['floors'])
        sqft_lot = float(request.POST['sqft_lot'])
        bathrooms = float(request.POST['bathrooms'])
        house_age = int(request.POST['house_age'])
        sqft_living15 = float(request.POST['sqft_living15'])
        lat = float(request.POST['lat'])
        long = float(request.POST['long'])
        
        features = np.array([[sqft_living, bedrooms,grade,floors,sqft_lot,bathrooms,house_age,sqft_living15,lat,long]])
        price =model.predict(features)[0]
        
    return render(request, 'predictor/index.html', {'price':price})