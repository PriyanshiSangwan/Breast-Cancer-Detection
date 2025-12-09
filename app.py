from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("breast_cancer_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Breast cancer feature names (30 features)
FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


@app.route("/")
def home():
    # Show empty form initially
    return render_template("index.html", feature_names=FEATURE_NAMES, prediction_text=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read values from the form
        input_values = []
        for i in range(len(FEATURE_NAMES)):
            value_str = request.form.get(f"feature_{i}", "").strip()
            if value_str == "":
                raise ValueError("All fields are required.")
            input_values.append(float(value_str))

        # Convert to numpy array and reshape
        input_array = np.asarray(input_values).reshape(1, -1)

        # Standardize using the saved scaler
        input_scaled = scaler.transform(input_array)

        # Predict using the loaded model
        prediction_probs = model.predict(input_scaled)  # shape (1, 2) with sigmoid
        predicted_class = int(np.argmax(prediction_probs, axis=1)[0])

        # Map class to label (same as your notebook)
        # 0 -> Malignant, 1 -> Benign
        if predicted_class == 0:
            result = "The Breast Cancer is Malignant"
        else:
            result = "The Breast Cancer is Benign"

        # Confidence of the predicted class
        confidence = float(np.max(prediction_probs) * 100)

        prediction_text = f"{result} (Model confidence: {confidence:.2f}%)"

    except ValueError as e:
        prediction_text = f"Error: {str(e)}. Please enter valid numeric values for all features."
    except Exception as e:
        prediction_text = f"Something went wrong: {str(e)}"

    # Send back the form + prediction
    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        prediction_text=prediction_text,
        previous_values=request.form,
    )


if __name__ == "__main__":
    # For local testing
    app.run(debug=True)
