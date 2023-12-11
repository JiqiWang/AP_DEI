#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        # predict
        y_hat = self.predict(x_i)
        # update
        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b

        label_scores = self.W.dot(x_i).reshape(-1,1)
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        
        # Softmax function
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # Make sure dimensions are compatible for element-wise operations      
        gradient = (y_one_hot - label_probabilities).reshape(-1, 1)
        # SGD update
        self.W += learning_rate * gradient * x_i

class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.w1 = np.random.normal(0.1, 0.01, (hidden_size, n_features))
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.normal(0.1, 0.01, ( n_classes, hidden_size))
        self.b2 = np.zeros((n_classes, 1))
    
    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, X):
        # Compute the forward pass of the network. At prediction timoye, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        # Forward pass
        z1 = self.w1.dot(X) + self.b1
        a1 = self.relu(z1)
        z2 = self.w2.dot(a1) + self.b2
        a2 = self.softmax(z2)
        return a1, a2

    def backward(self, X, y, a1, a2, learning_rate=0.001):
        # One-hot encoding of the target label
        y_one_hot = np.zeros(self.b2.shape)
        y_one_hot[y] = 1

        # Backward pass
        # Compute gradients for the output layer
        dz2 = a2 - y_one_hot
        dw2 = dz2.dot(a1.T)

        # Compute gradients for the hidden layer
        da1 = np.dot(self.w2.T, dz2)
        dz1 = da1 * (a1 > 0)  # ReLU derivative
        dw1 = dz1.dot(X.T) 

        # Update weights and biases
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * dz2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * dz1

        # Compute the loss
        loss = -np.sum(y_one_hot * np.log(a2))
        return loss
    
    def predict(self, X):
        _, a2 = self.forward(X.T)
        return np.argmax(a2, axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        total_loss = 0

        for x_i, y_i in zip(X, y):
            x_i = x_i.reshape((-1, 1))
            a1, a2 = self.forward(x_i)
            loss = self.backward(x_i, y_i, a1, a2, learning_rate)
            total_loss += loss
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(X)

        return avg_loss


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.savefig('output11.png')
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.savefig('outputloss.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
