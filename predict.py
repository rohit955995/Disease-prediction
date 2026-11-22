import joblib
import numpy as np

def load_artifacts(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(input_data, model, scaler):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

if __name__ == "__main__":
    model, scaler = load_artifacts('../models/final_model.pkl', '../models/scaler.pkl')
    example_patient = [63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0]
    pred, prob = predict(example_patient, model, scaler)
    print(f"Prediction: {'Disease' if pred == 1 else 'No Disease'}")
    print(f"Probability of Disease: {prob:.4f}")
