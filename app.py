import os
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("reg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for JSON API prediction (for Postman or frontend AJAX)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']  # Expecting JSON: {"data": {"feature1": val, ...}}
    # Convert input to proper shape and scale
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    # Make prediction
    output = model.predict(new_data)
    # Return result as JSON
    return jsonify({"prediction": float(output[0])})

# Route for HTML form prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    data = [float(x) for x in request.form.values()]
    # Scale input
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    # Make prediction
    output = model.predict(final_input)[0]
    # Render home page with prediction text
    return render_template("home.html", prediction_text=f"The predicted price for the house is {output:.2f}")

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
