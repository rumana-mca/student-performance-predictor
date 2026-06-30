import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# load dataset
df = pd.read_csv("student_data.csv")

np.random.seed(42)
noise = np.random.normal(0, 4.5, size=len(df))
df["FinalMarks"] = df["FinalMarks"] + noise
df["FinalMarks"] = df["FinalMarks"].clip(0, 100)


# inputs and output
X = df[["StudyHours", "Attendance", "SleepHours", "PreviousMarks", "AssignmentsScore", "Participation" ]]
y = df["FinalMarks"]

# model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# train
model.fit(X_train, y_train)
print("Coefficients:")
for name, coef in zip(X_train.columns, model.coef_):
    print(name, coef)
# prediction (example)
new_data = pd.DataFrame([[5, 80, 7, 60, 65, 60]],
columns=["StudyHours", "Attendance", "SleepHours", "PreviousMarks", "AssignmentsScore", "Participation"])

prediction = model.predict(new_data)

print("Predicted Marks:", prediction)