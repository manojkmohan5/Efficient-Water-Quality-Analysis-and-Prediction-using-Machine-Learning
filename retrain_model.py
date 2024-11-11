# retrain_model.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load your dataset
# Replace 'your_dataset.csv' with the actual path to your dataset
data = pd.read_csv('/Users/pranavparthasarathy/Downloads/Project_demo/water_potability.csv')
print(data.columns)
# Set the feature columns and target column
feature_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
target_column = 'Potability'


X = data[feature_columns]
y = data[target_column]

# Split data into training and testing sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, '/Users/pranavparthasarathy/Downloads/Project_demo/wqi.joblib')

print("Model retrained and saved successfully.")
