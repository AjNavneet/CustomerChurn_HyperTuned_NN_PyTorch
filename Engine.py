import pandas as pd
from MLPipeline.Preprocessing import Preprocessing
from MLPipeline.SaveAndLoad import SaveAndLoad
from MLPipeline.Dropout import DropoutLayer
from MLPipeline.EarlyStopping import EarlyStopping
from MLPipeline.NeuralNet import NeuralNet
from MLPipeline.Regularization import Regularization

# Reading the data from a CSV file
df = pd.read_csv("Input/data.csv")

# Initialize the Preprocessing object with the dataframe
data = Preprocessing(df)

# Drop specific columns ('customer_id', 'phone_no', 'year')
data = data.drop(["customer_id", "phone_no", "year"])

# Drop rows with missing values (null values)
data = data.dropna()

# Scale numerical features
data = data.scale()

# Encode categorical features
data = data.encode()

# Perform SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance
target_col = 'churn'  # Specify the target column name
x_smote, y_smote = data.smote(target_col)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = data.split_data(x_smote, y_smote)

# Initialize the data loader
loader = data.data_loader(X_train, y_train)

# Create and train a basic neural network
NeuralNet(X_train, y_train, X_test, y_test, loader)

# Create and train a neural network with dropout layers
DropoutLayer(X_train, y_train, X_test, y_test, loader)

# Create and train a neural network with regularization
Regularization(X_train, y_train, X_test, y_test, loader)

# Create and train a neural network with early stopping
EarlyStopping(X_train, y_train, X_test, y_test, loader)

# Save and load models
SaveAndLoad(X_train, y_train, X_test, y_test, loader)
