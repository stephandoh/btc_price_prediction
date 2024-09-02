from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('crypto_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1, -1)
    
    # Scale the features using the loaded scaler
    final_features_scaled = scaler.transform(final_features)
    
    # Make the prediction using the loaded model
    prediction = model.predict(final_features_scaled)
    
    # Round the prediction to 5 decimal places
    output = round(prediction[0], 5)

    return render_template('index.html', prediction_text=f'Predicted BTC closing price is ${output}')

if __name__ == "__main__":
    app.run(debug=True)
