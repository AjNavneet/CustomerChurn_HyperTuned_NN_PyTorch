# Importing necessary libraries
import numpy as np
from sklearn.preprocessing import LabelEncoder  # For label encoding of categorical features
from imblearn.over_sampling import SMOTE  # For Synthetic Minority Over-sampling Technique
from torch import Tensor  # PyTorch tensor data structure
import torch.utils.data as Data  # PyTorch data loading utilities
from sklearn.preprocessing import MinMaxScaler  # For feature scaling

# Definition of the Preprocessing class
class Preprocessing:

    def __init__(self, data):
        # Constructor to initialize the class with input data
        self.data = data

    # Method to drop specified columns from the data
    def drop(self, cols):
        """
        Drop specified columns from the data.
        :param cols: List of column names to be dropped
        :return: Data with specified columns removed
        """
        col = list(cols)
        self.data.drop(col, axis=1, inplace=True)
        return self.data

    # Method to drop rows with null values
    def dropna(self):
        """
        Drop rows with null values from the data.
        :return: Data with rows containing null values removed
        """
        self.data.dropna(axis=0, inplace=True)
        return self.data

    # Method to scale numerical features using Min-Max scaling
    def scale(self):
        """
        Scale numerical features using Min-Max scaling.
        :return: Data with numerical features scaled
        """
        num_cols = self.data.select_dtypes(exclude=['object']).columns.tolist()  # Get numerical columns
        scale = MinMaxScaler()
        self.data[num_cols] = scale.fit_transform(self.data[num_cols])
        return self.data

    # Method to perform label encoding on categorical features
    def encode(self):
        """
        Perform label encoding on categorical features.
        :return: Data with categorical features label encoded
        """
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()  # Get categorical columns
        le = LabelEncoder()
        self.data[cat_cols] = self.data[cat_cols].apply(le.fit_transform)
        return self.data

    # Method to apply Synthetic Minority Over-sampling Technique (SMOTE) for handling class imbalance
    def smote(self, target_col):
        """
        Apply Synthetic Minority Over-sampling Technique (SMOTE) for handling class imbalance.
        :param target_col: Name of the target column
        :return: Resampled predictor variables (X) and target variable (Y)
        """
        smote = SMOTE()
        x_smote, y_smote = smote.fit_resample(self.data.drop(target_col, axis=1), self.data[target_col])
        return x_smote, y_smote

    # Method to split data into training and testing sets
    def split_data(self, X, Y):
        """
        Split data into training and testing sets.
        :param X: Predictor variables
        :param Y: Target variable
        :return: Training and testing sets for X and Y
        """
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.2, random_state=42, stratify=Y)
        return X_train, X_test, y_train, y_test

    # Method to create a data loader for batch processing
    def data_loader(self, X_train, y_train):
        """
        Create a data loader for batch processing.
        :param X_train: Training data (predictor variables)
        :param y_train: Training data (target variable)
        :return: Data loader
        """
        X_train = Tensor(X_train)
        y_train = Tensor(np.array(y_train)

        BATCH_SIZE = 64  # Batch size for training

        torch_dataset = Data.TensorDataset(X_train, y_train)

        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Number of subprocesses to use for data loading
        )
        return loader
