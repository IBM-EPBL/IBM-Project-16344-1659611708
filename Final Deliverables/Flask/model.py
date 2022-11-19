import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("C:/Users/sowmi/Downloads/crop_production.csv")
df[:5]

# Droping Nan Values
data = df.dropna()
test = df[~df["Production"].notna()].drop("Production",axis=1)

sum_maxp = data["Production"].sum()
data["percent_of_production"] = data["Production"].map(lambda x:(x/sum_maxp)*100)

data1 = data.drop(["District_Name","Crop_Year"],axis=1)

features = pd.get_dummies(data1)

labels = np.array(features['Production'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Production', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 5, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels);

filename = 'model_final.sav'
joblib.dump(rf, filename)