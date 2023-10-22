# Importing necessary libraries
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch import optim
import torch
from torch import nn
from torch import optim
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch import Tensor
import numpy as np

# Definition of the SaveAndLoad class
class SaveAndLoad:

    def __init__(self, X_train, y_train, X_test, y_test, loader):
        # Constructor to initialize the class with data and loader

        model_chk = self.model_arch(X_train.shape[1])  # Create the model architecture
        print(model_chk)
        optim, path, torch = self.train(model_chk, X_train, loader)  # Train the model
        self.evaluate(model_chk, optim, path, torch, X_test, y_test)  # Evaluate the model

    # Method to evaluate the model on test data
    def evaluate(self, model_chk, optim, path, torch, X_test, y_test):
        """
        Evaluate the model.
        :param model_chk: Model to be evaluated
        :param optim: Optimizer
        :param path: Path to save and load model checkpoints
        :param torch: Torch library
        :param X_test: Test data (predictor variables)
        :param y_test: Test data (target variable)
        """
        model_load = model_chk
        optimizer = optim.Adam(model_load.parameters(), lr=1e-4, weight_decay=1e-5)
        checkpoint = torch.load(path + "model_2.pt")
        model_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model_load.eval()
        # Converting data into tensor form
        X_test_tensor = Tensor(X_test)
        y_test = Tensor(np.array(y_test))
        z = model_load(X_test_tensor)
        yhat = list(z.argmax(1))  # Get predicted values
        y_test = list(y_test)
        print("Accuracy Score of Test Data is:", round(accuracy_score(y_test, yhat) * 100, 2), "%")

    # Method to train the model
    def train(self, model_chk, X_train, loader):
        """
        Train the model.
        :param model_chk: Model to be trained
        :param X_train: Training data (predictor variables)
        :param loader: Data loader for batch processing
        :return: Optimizer, path to checkpoints, torch library
        """
        # Define the loss
        criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        # Regularization
        # Optimizers require the parameters to optimize and a learning rate
        # Add L2 regularization to the optimizer by specifying weight_decay
        optimizer = optim.Adam(model_chk.parameters(), lr=1e-4, weight_decay=1e-5)
        epochs = 3
        path = "Output/model/"

        for e in range(epochs):
            running_loss = 0
            for step, (batch_x, batch_y) in enumerate(loader):

                b_x = Variable(batch_x)
                b_y = Variable(batch_y.type(torch.LongTensor))

                # Training pass
                optimizer.zero_grad()

                output = model_chk(b_x)
                loss = criterion(output, b_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                torch.save({
                    'epoch': e,
                    'model_state_dict': model_chk.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                }, path + "model_" + str(e) + ".pt")
            else:
                print(f"Training loss: {running_loss / len(X_train)}")
        return optim, path, torch

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
        model_chk = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
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
        return model_chk
