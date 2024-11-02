#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load datasets
normalskin_df = pd.read_csv("normalskin_.csv")
cancerskin_df = pd.read_csv("cancerskin_.csv")

# Add labels for classification (0 for normal skin, 1 for cancerous skin)
normalskin_df['Label'] = 0
cancerskin_df['Label'] = 1

# Combine datasets
data = pd.concat([normalskin_df, cancerskin_df], ignore_index=True)

# Separate features and label
X = data.drop(columns=['Label'])
y = data['Label']

# Handle NaN values by imputing with the median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Add Gaussian noise to each feature column to reduce overfitting
noise_factor = 2.2 # Increase noise level to reduce accuracy
X_noisy = X_imputed + noise_factor * np.random.normal(size=X_imputed.shape)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_noisy)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a simpler classifier with reduced complexity (e.g., fewer trees, limited depth)
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Perform cross-validation to evaluate generalization
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")


# In[ ]:




