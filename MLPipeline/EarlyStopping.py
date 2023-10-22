# Importing necessary libraries
from torch import nn  # Neural network module in PyTorch
from torch import optim  # Optimization module in PyTorch
import torch  # PyTorch library for deep learning
from torch.autograd import Variable  # Variables for automatic differentiation
from sklearn.metrics import accuracy_score  # For calculating accuracy
from torch import Tensor  # Tensor data structure in PyTorch
import numpy as np  # Numerical Python library for numerical operations

# Definition of the EarlyStopping class
class EarlyStopping:

    def __init__(self, X_train, y_train, X_test, y_test, loader):
        # Initialize the model with architecture based on input data shape
        model_early_stp = self.model_arch(X_train.shape[1])
        
        # Train the model with training data
        self.train(model_early_stp, X_train, loader)
        
        # Evaluate the model using test data
        self.evaluate(model_early_stp, X_test, y_test)
    
    # Evaluating the model on test data
    def evaluate(self, model_early_stp, X_test, y_test):
        """
        Evaluating the model
        :param model_early_stp: Trained neural network model
        :param X_test: Test data
        :param y_test: True labels of the test data
        """
        # Convert the test data to PyTorch tensors
        X_test_tensor = Tensor(X_test)
        y_test = Tensor(np.array(y_test))
        
        # Forward pass through the model
        z = model_early_stp(X_test_tensor)
        
        # Get the predicted class labels
        yhat = list(z.argmax(1))
        y_test = list(y_test)
        
        # Calculate and print the accuracy score
        accuracy = round(accuracy_score(y_test, yhat) * 100, 2)
        print("Accuracy Score of Test Data is ", accuracy, "%")
    
    # Train the model
    def train(self, model_early_stp, X_train, loader):
        """
        Training the model
        :param model_early_stp: Neural network model
        :param X_train: Training data
        :param loader: Data loader for batch processing
        """
        # Define the loss function (Negative Log-Likelihood Loss)
        criterion = nn.NLLLoss()
        
        # Define the optimizer (Adam optimizer with L2 weight decay)
        optimizer = optim.Adam(model_early_stp.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Number of training epochs
        epochs = 100
        epochs_no_improve = 0
        early_stop = False
        min_loss = np.Inf
        iter = 0

        for e in range(epochs):
            running_loss = 0
            if early_stop:
                print("Stopped")
                break
            else:
                for step, (batch_x, batch_y) in enumerate(loader):

                    b_x = Variable(batch_x)
                    b_y = Variable(batch_y.type(torch.LongTensor))

                    # Training pass
                    optimizer.zero_grad()

                    output = model_early_stp(b_x)
                    loss = criterion(output, b_y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if abs(running_loss) < abs(min_loss):
                        epochs_no_improve = 0
                        min_loss = running_loss
                    else:

                        epochs_no_improve += 1
                    iter += 1

                    if e > 5 and epochs_no_improve == epochs:
                        print('Early stopping!')
                        early_stop = True
                        break
                    else:
                        continue

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
        model_early_stp = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                        nn.Dropout(0.5),  # 50% Probability dropout
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                        torch.nn.Dropout(0.2),  # 20% Probability dropout
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[1], hidden_sizes[1]),
                                        torch.nn.Dropout(0.1),  # 10% Probability dropout
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[1], output_size),
                                        nn.Softmax(dim=1))  # Softmax activation for multi-class classification

        print(model_early_stp)
        return model_early_stp
