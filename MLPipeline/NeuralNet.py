# Importing necessary libraries
from torch import nn  # Neural network module in PyTorch
from torch import optim  # Optimization module in PyTorch
import torch  # PyTorch library for deep learning
from sklearn.metrics import accuracy_score  # For calculating accuracy
from torch.autograd import Variable  # Variables for automatic differentiation
from torch import Tensor  # Tensor data structure in PyTorch
import numpy as np  # Numerical Python library for numerical operations

# Definition of the NeuralNet class
class NeuralNet:

    def __init__(self, X_train, y_train, X_test, y_test, loader):
        # Initialize the model with architecture based on input data shape
        model = self.model_arch(X_train.shape[1])
        
        # Train the model with training data
        self.train(model, X_train, loader)
        
        # Evaluate the model using test data
        self.evaluate(model, X_test, y_test)

    # Evaluate the model on test data
    def evaluate(self, model, X_test, y_test):
        """
        Evaluate the model
        :param model: Trained neural network model
        :param X_test: Test data
        :param y_test: True labels of the test data
        """
        # Convert the test data to PyTorch tensors
        X_test_tensor = Tensor(X_test)
        y_test = Tensor(np.array(y_test))
        
        # Forward pass through the model
        z = model(X_test_tensor)
        
        # Get the predicted class labels
        yhat = list(z.argmax(1))
        y_test = list(y_test)
        
        # Calculate and print the accuracy score
        accuracy = round(accuracy_score(y_test, yhat) * 100, 2)
        print("Accuracy Score of Test Data is:", accuracy, "%")

    # Train the model
    def train(self, model, X_train, loader):
        """
        Train the model
        :param model: Neural network model
        :param X_train: Training data
        :param loader: Data loader for batch processing
        """
        # Define the loss function (Negative Log-Likelihood Loss)
        criterion = nn.NLLLoss()
        
        # Define the optimizer (Stochastic Gradient Descent)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Number of training epochs
        epochs = 10
        for e in range(epochs):
            running_loss = 0
            for step, (batch_x, batch_y) in enumerate(loader):

                b_x = Variable(batch_x)
                b_y = Variable(batch_y.type(torch.LongTensor))

                # Training pass
                optimizer.zero_grad()

                output = model(b_x)
                loss = criterion(output, b_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                print(f"Training loss: {running_loss / len(X_train)}")

    # Define the model architecture
    def model_arch(self, input_size):
        """
        Model Architecture
        :return: Neural network model
        """
        # Define hidden layer sizes
        hidden_sizes = [128, 64]
        output_size = 2
        
        # Build a feed-forward neural network model using PyTorch's Sequential
        model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[1], output_size),
                              nn.Softmax(dim=1))  # Softmax activation for multi-class classification

        print(model)
        return model
