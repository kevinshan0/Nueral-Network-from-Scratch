import numpy as np
import csv
import ast

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Cross-entropy loss function and its derivative
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / y_true.shape[0]

def cross_entropy_loss_derivative(y_true, y_pred):
    return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

# Neural network class
def initialize_network(input_size=81, hidden_layers=[128, 128], output_size=81):
    np.random.seed(42)
    network = {}

    # Initialize weights and biases for the first layer
    network["W1"] = np.random.randn(input_size, hidden_layers[0]) * 0.1
    network["b1"] = np.zeros((1, hidden_layers[0]))

    # Initialize weights and biases for hidden layers
    for i in range(1, len(hidden_layers)):
        network[f"W{i+1}"] = np.random.randn(hidden_layers[i-1], hidden_layers[i]) * 0.1
        network[f"b{i+1}"] = np.zeros((1, hidden_layers[i]))

    # Initialize weights and biases for the output layer
    network[f"W{len(hidden_layers)+1}"] = np.random.randn(hidden_layers[-1], output_size) * 0.1
    network[f"b{len(hidden_layers)+1}"] = np.zeros((1, output_size))

    return network

def forward_propagation(network, X):
    activations = {"A0": X}
    Z = {}

    # Forward propagate through each layer
    num_layers = len(network) // 2
    for i in range(1, num_layers + 1):
        Z[f"Z{i}"] = np.dot(activations[f"A{i-1}"], network[f"W{i}"]) + network[f"b{i}"]
        activations[f"A{i}"] = sigmoid(Z[f"Z{i}"])

    return Z, activations

def back_propagation(network, X, y, Z, activations, learning_rate=0.01):
    m = X.shape[0]
    num_layers = len(network) // 2
    gradients = {}

    # Compute gradients for the output layer
    dZ = activations[f"A{num_layers}"] - y
    gradients[f"dW{num_layers}"] = np.dot(activations[f"A{num_layers-1}"].T, dZ) / m
    gradients[f"db{num_layers}"] = np.sum(dZ, axis=0, keepdims=True) / m

    # Backpropagate through hidden layers
    for i in range(num_layers - 1, 0, -1):
        dA = np.dot(dZ, network[f"W{i+1}"].T)
        dZ = dA * sigmoid_derivative(Z[f"Z{i}"])
        gradients[f"dW{i}"] = np.dot(activations[f"A{i-1}"].T, dZ) / m
        gradients[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m

    # Update weights and biases
    for i in range(1, num_layers + 1):
        network[f"W{i}"] -= learning_rate * gradients[f"dW{i}"]
        network[f"b{i}"] -= learning_rate * gradients[f"db{i}"]

def train_network(network, X, y, hidden_layers, epochs=1000, learning_rate=0.01):
    for epoch in range(epochs):
        Z, activations = forward_propagation(network, X)
        loss = cross_entropy_loss(y, activations[f"A{len(hidden_layers)+1}"])
        back_propagation(network, X, y, Z, activations, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

def predict(network, X):
    _, activations = forward_propagation(network, X)
    return np.round(activations[f"A{len(network)//2}"])

# Function to parse training_data.csv
def parse_csv(file_path):
    data = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                # Safely parse the input and output arrays
                input_array = np.array(ast.literal_eval(row[0])).flatten()
                output_array = np.array(ast.literal_eval(row[1])).flatten()
                data.append([input_array, output_array])
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing row: {row}. Skipping. Error: {e}")
                continue

    # Convert to a NumPy array with shape (rows, 2)
    return np.array(data, dtype=object)

# Example usage
if __name__ == "__main__":
    # Parse the CSV file
    file_path = "training_data.csv"
    parsed_data = parse_csv(file_path)
    print("Parsed Data Shape:", parsed_data.shape)
    print("Sample Row:", parsed_data[0])

    # Separate inputs (X) and outputs (Y)
    X = np.array([row[0] for row in parsed_data])
    Y = np.array([row[1] for row in parsed_data])

    # Split into training and testing data
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    print("X_train Shape:", X_train.shape)
    print("Y_train Shape:", Y_train.shape)
    print("X_test Shape:", X_test.shape)
    print("Y_test Shape:", Y_test.shape)

    # Initialize network with adjustable hidden layers
    hidden_layers = [128, 256, 128]
    network = initialize_network(hidden_layers=hidden_layers)

    # Train network
    train_network(network, X_train, Y_train, hidden_layers, epochs=1000, learning_rate=0.01)

    # Test prediction
    predictions = predict(network, X_test)
    print("Predictions:", predictions)
