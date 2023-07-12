from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("new_dataset.csv")
data = np.array(df)
X = np.delete(data[1:, :], 3, axis=1)  # Remove the 4th column from the array
y = data[1:, 3]  # Select the 4th column as the target variable

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y)

# Create and train the models
#lin_reg = LinearRegression().fit(x_train, y_train)
#log_reg = LogisticRegression().fit(x_train, y_train)
svc_model = SVC().fit(x_train, y_train)

# Save the trained models using pickle
#pickle.dump(lin_reg, open('lin_model.pkl', 'wb'))
#pickle.dump(log_reg, open('log_model.pkl', 'wb'))
pickle.dump(svc_model, open('svc_model.pkl', 'wb'))
