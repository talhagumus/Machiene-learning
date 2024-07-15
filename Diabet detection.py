import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Load the data
data = pd.read_csv('diabetes.csv')

# Drop the 4th row, as it will be used to test the model
data = data.drop(4)

# Separate the input and output variables
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=0)

# Create and train the model
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

# Make predictions on the test data and print the accuracy score
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(f"Accuracy score of the test data: {test_data_accuracy}")

# Get the input data(4. row)
input_data = (0, 137, 40, 35, 168, 43.1, 2.288, 33)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped_array = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_input_data = scaler.transform(input_data_reshaped_array)

# Make a prediction
prediction = model.predict(std_input_data)
if prediction[0] == 0:
    print("Person is not diabetic")
if prediction[0] == 1:
    print("Person is diabetic")
  


