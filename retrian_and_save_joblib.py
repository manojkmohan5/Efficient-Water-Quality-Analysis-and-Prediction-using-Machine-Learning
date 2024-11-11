import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Corrected import
import joblib  # Use joblib for saving models

# Load your dataset
data = pd.read_csv('/Users/pranavparthasarathy/Downloads/Project_demo/water_potability.csv')

# Define feature columns and target column
feature_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
target_column = 'Potability'

# Split data into features and target
X = data[feature_columns]
y = data[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, shuffle=True)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, 'wqi_model.joblib')

print("Model trained and saved successfully.")
