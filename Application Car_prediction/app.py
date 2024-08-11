from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    make = int(request.form['make'])
    model_ = int(request.form['model'])
    year = int(request.form['year'])
    mileage = int(request.form['mileage'])
    condition = int(request.form['condition'])
    car_age = int(request.form['car_age'])
    make_model = int(request.form['make_model'])
    condition_numeric = int(request.form['condition_numeric'])
    mileage_bracket = int(request.form['mileage_bracket'])
    
    # Create a feature array for prediction
    features = np.array([[make, model_, year, mileage, condition, car_age, make_model, condition_numeric, mileage_bracket]])
    
    # Predict the price
    prediction = model.predict(features)
    
    # Print the prediction in the console
    print(f'Predicted Price: £{prediction[0]:.2f}')
    
    # Render the prediction result in the template
    return render_template('index.html', prediction_text=f'Predicted Price: £{prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
