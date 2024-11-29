from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load models
models = {
    "diabetes": pickle.load(open("saved_models/diabetes_disease_model.sav", "rb")),
    "heart": pickle.load(open("saved_models/heart_disease_model.sav", "rb")),
    "parkinsons": pickle.load(open("saved_models/parkinsons_disease_model.sav", "rb"))
}

@app.route('/predict/<disease_type>', methods=['POST'])
def predict(disease_type):
    data = request.json
    model = models.get(disease_type)
    
    if not model:
        return jsonify({"error": "Invalid disease type"}), 400

    input_data = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(input_data)
    
    message = "Positive" if prediction[0] == 1 else "Negative"
    return jsonify({"message": f"The result is {message}."})

if __name__ == '__main__':
    app.run(debug=True)
