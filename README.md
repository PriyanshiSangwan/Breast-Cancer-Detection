Breast Cancer Detection using Deep Learning

A Deep Learning-based web application that predicts whether a breast tumor is Malignant (cancerous) or Benign (non-cancerous).
The goal of this project is to support early breast cancer diagnosis using automated prediction models.

## Live Deployment:
https://breast-cancer-detection-priyanshi-my7c.onrender.com

## Project Features

✔ Predicts cancer type using medical biopsy data
✔ Compares multiple ML classifiers
✔ Best model selected automatically
✔ Simple and interactive web-based UI
✔ High accuracy for clinical decision support

## Dataset Information

Source: Scikit-learn Breast Cancer Wisconsin Dataset

Total Samples: 569
Classes:

0 → Malignant

1 → Benign

Each sample contains 30 medical features such as:
- Radius
- Texture
- Smoothness
- Compactness
- Concavity, etc.

## Tools & Technologies Used
Category	Technology
Programming	Python
ML Libraries	NumPy, Pandas, Scikit-Learn
Visualization	Matplotlib, Seaborn
Model Deployment	Flask
Hosting	Render
## Machine Learning Models Evaluated
Model	Accuracy
Logistic Regression	~76%
Decision Tree Classifier	~96%
XGBoost Classifier	~97%
Random Forest Classifier	98.82% (Best Model)

The Random Forest Classifier is selected for final deployment.

## Workflow of the Project

Import & explore the dataset

Data cleaning (null checks, processing & scaling)

EDA & feature visualization

Train-Test split

Train multiple ML algorithms

Performance evaluation

Best model selection

Model saving using joblib/pickle

Web-based deployment with Flask & Render

📁 Project Structure
📦 Breast Cancer Detection
├── app.py                # Flask web application
├── model.pkl             # Saved best ML model
├── index.html            # Frontend UI
├── requirements.txt      # Dependencies list
└── README.md             # Project documentation



##  Business/Medical Use Case

Helps radiologists in quick diagnosis

Supports early decision-making

Useful for hospitals, clinics, healthcare AI systems

##### Disclaimer: This project is for educational purposes only and not a substitute for clinical diagnosis.

## Model Output

Prediction displayed as:

Benign → Non-cancerous tumor

Malignant → Cancerous tumor

## Conclusion

The model achieved excellent performance with 98.82% accuracy using Random Forest.
This demonstrates how AI can greatly improve cancer detection and save lives through early screening.

 ## Author

Priyanshi Sangwan
BTech CSE 
3rd Year student
ML and Data Science Enthusiast
E-mail: priyanshisangwan38@gmail
Linkedin: https://www.linkedin.com/in/priyanshi-sangwan-4782992a5/
