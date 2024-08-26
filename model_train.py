import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load your data
data = pd.read_csv('BitcoinHeistData.csv')

# Preprocess the labels
data['label'] = data['label'].apply(lambda x: 'Ransomware' if x != 'white' else 'white')

# Drop the 'address' column
data = data.drop(columns=['address'])

# Separate majority and minority classes
white = data[data['label'] == 'white']
ransomware = data[data['label'] == 'Ransomware']

# Resample to balance the classes
ransomware_upsampled = resample(ransomware, 
                                replace=True,     # sample with replacement
                                n_samples=len(white),    # to match majority class
                                random_state=123) # reproducible results

# Combine majority class with upsampled minority class
data_balanced = pd.concat([white, ransomware_upsampled])

# Split the data into features and labels
X = data_balanced.drop('label', axis=1)
y = data_balanced['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=6)  # Adjust n_components as necessary
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Build the model
model = Sequential()
model.add(Dense(128, input_dim=X_train_pca.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model with class weights
class_weight = {0: 1., 1: (len(y_encoded) / sum(y_encoded))}  # Adjust this weight based on class distribution
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train_pca, y_train, validation_split=0.2, epochs=60, batch_size=64, class_weight=class_weight, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pca, y_test)
print(f'Accuracy: {accuracy:.4f}')

# Predictions and detailed metrics
from sklearn.metrics import classification_report

y_pred = (model.predict(X_test_pca) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the trained model, scaler, label encoder, and PCA
model.save('deep_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

print("Model, PCA, scaler, and label encoder saved successfully.")
