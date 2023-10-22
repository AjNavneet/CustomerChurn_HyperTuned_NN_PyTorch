# Importing necessary libraries
from torch import nn
from torch import optim
import torch
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch import Tensor
import numpy as np

# Definition of the Regularization class
class Regularization:

    def __init__(self, X_train, y_train, X_test, y_test, loader):
        # Constructor to initialize the class with data and loader

        model_reg = self.model_arch(X_train.shape[1])  # Create the model architecture

        self.train(model_reg, X_train, loader)  # Train the model

        self.evaluate(model_reg, X_test, y_test)  # Evaluate the model

    # Method to evaluate the model on test data
    def evaluate(self, model_reg, X_test, y_test):
        """
        Evaluate the model on test data.
        :param model_reg: Model to be evaluated
        :param X_test: Test data (predictor variables)
        :param y_test: Test data (target variable)
        """
        # Converting data into tensor form
        X_test_tensor = Tensor(X_test)
        y_test = Tensor(np.array(y_test))
        z = model_reg(X_test_tensor)
        yhat = list(z.argmax(1))  # Get predicted values
        y_test = list(y_test)
        print("Accuracy Score of Test Data is", round(accuracy_score(y_test, yhat) * 100, 2), "%")

    # Method to train the model
    def train(self, model_reg, X_train, loader):
        """
        Train the model.
        :param model_reg: Model to be trained
        :param X_train: Training data (predictor variables)
        :param loader: Data loader for batch processing
        """
        # Define the loss
        criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        # Regularization
        # Optimizers require the parameters to optimize and a learning rate
        # Add L2 regularization to the optimizer by specifying weight_decay
        optimizer = optim.Adam(model_reg.parameters(), lr=1e-4, weight_decay=1e-5)
        epochs = 10
        for e in range(epochs):
            running_loss = 0
            for step, (batch_x, batch_y) in enumerate(loader):

                b_x = Variable(batch_x)
                b_y = Variable(batch_y.type(torch.LongTensor))

                # Training pass
                optimizer.zero_grad()

                output = model_reg(b_x)
                loss = criterion(output, b_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                print(f"Training loss: {running_loss / len(X_train)}")

    # Method to define the model architecture
    def model_arch(self, input_size):
        """
        Define the model architecture.
        :param input_size: Number of input features
        :return: Model with specified architecture
        """
        # Creating a network with dropout layers
        hidden_sizes = [128, 64]
        output_size = 2
        # Build a feed-forward network
        model_reg = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                  nn.Dropout(0.5),  # 50% Probability
                                  nn.ReLU(),
                                  nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                  torch.nn.Dropout(0.2),  # 20% Probability
                                  nn.ReLU(),
                                  nn.Linear(hidden_sizes[1], hidden_sizes[1]),
                                  torch.nn.Dropout(0.1),  # 10% Probability
                                  nn.ReLU(),
                                  nn.Linear(hidden_sizes[1], output_size),
                                  nn.Softmax(dim=1))
        print(model_reg)
        return model_reg
