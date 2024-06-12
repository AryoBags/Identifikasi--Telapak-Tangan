import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data preprocessing
data = pd.read_csv('Tangan-Kiri-Canny-Hue_semua-perbaikan.csv')

# Display the first few rows of the dataset
print(data.head())

# Separate the features and the target variable
X = data[['Number of Lines Detected', 'Number of Edges Detected by Canny', 'Hu Moment 1', 'Hu Moment 2', 
          'Hu Moment 3', 'Hu Moment 4', 'Hu Moment 5', 'Hu Moment 6', 'Hu Moment 7']]
y = data['Label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network model
model = MLPClassifier(hidden_layer_sizes=(175, 125, 75), max_iter=500, activation='relu', solver='adam', random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and the scaler
pickle.dump(model, open("model-kiri.pkl", 'wb'))
pickle.dump(scaler, open("scaler.pkl", 'wb'))
