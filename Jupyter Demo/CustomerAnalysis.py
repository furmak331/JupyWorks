
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Step 1: Load Dataset
# Assume the dataset is 'customer_transactions.csv' with columns: ['CustomerID', 'Age', 'Gender', 'AnnualIncome', 'SpendingScore', 'PurchaseFrequency']
data = pd.read_csv('customer_transactions.csv')

# Step 2: Data Preprocessing
# Handling missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical data
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Feature scaling
scaler = StandardScaler()
data[['Age', 'AnnualIncome', 'SpendingScore', 'PurchaseFrequency']] = scaler.fit_transform(data[['Age', 'AnnualIncome', 'SpendingScore', 'PurchaseFrequency']])

# Step 3: Exploratory Data Analysis (EDA)
# Visualize the correlation matrix
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of Spending Scores
sns.histplot(data['SpendingScore'], bins=20, kde=True)
plt.title('Distribution of Spending Scores')
plt.show()

# Step 4: Advanced Data Analysis - Clustering to find unique patterns
# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[['Age', 'AnnualIncome', 'SpendingScore', 'PurchaseFrequency']])
data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(data[['PCA1', 'PCA2']])

# Visualize the clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='Set2')
plt.title('Customer Segmentation based on Clustering')
plt.show()

# Step 5: Feature Selection - Identifying unique patterns
# Check feature importance using RandomForest
X = data[['Age', 'AnnualIncome', 'SpendingScore', 'PurchaseFrequency']]
y = data['Cluster']

model = RandomForestClassifier()
model.fit(X, y)

# Display feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
print("Feature Importance:\n", feature_importance)

# Step 6: Modeling - Predictive analysis to identify loyal customers
# Create a new binary column 'Loyalty' based on PurchaseFrequency threshold
data['Loyalty'] = np.where(data['PurchaseFrequency'] > 0.5, 1, 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, data['Loyalty'], test_size=0.3, random_state=42)

# Train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = rf_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize Feature Importance
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Loyalty Prediction')
plt.show()
