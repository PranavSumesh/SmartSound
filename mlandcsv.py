import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset from CSV
df = pd.read_csv("data.csv")

# Preprocessing
df["Time"] = df["Time"].apply(lambda x: int(x.split(".")[0].split(":")[0]))  # Extract hour from time
df["Day"] = df["Day"].replace({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7})  # Map days of the week to numerical values

# Split features and target variable
X = df[["Time", "Day"]]
y = df["Volume"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree regression model with hyperparameter tuning
model = DecisionTreeRegressor(max_depth=5, min_samples_split=5, min_samples_leaf=2)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plotting
plt.scatter(X_test["Time"], y_test, color='darkorange', label="Actual")
plt.scatter(X_test["Time"], y_pred, color='blue', label="Predicted")
plt.xlabel("Time")
plt.ylabel("Volume")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

# Prediction for 6 PM on Wednesday
predicted_volume = model.predict([[18, 3]])
print("Predicted volume for 6 PM on Wednesday:", predicted_volume)
