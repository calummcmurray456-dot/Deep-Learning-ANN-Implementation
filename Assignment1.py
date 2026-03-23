import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#using a standard value for 42 as the random seed for reproducibility
np.random.seed(42)

#Task1

#loading the dataset
data = np.loadtxt('german_credit_simplified.txt')

#features normalised
X = data[:, :24]
y = data[:, 24].reshape(-1, 1)

#splitting the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_input = X_train.shape[1]
# 16 neurons for hidden layer chosen as a balance between model capacity and 
# overfitting risk. For 24 input features with binary classification, 16 
# neurons (~0.67x inputs) provides sufficient representational power whilst 
# keeping the network small enough to avoid overfitting on 800 training samples.
n_hidden = 16
n_output = 1

#weights randomly initialised uniformly from -0.1 to 0.1
#biases initialised as 0
W1 = np.random.uniform(-0.1, 0.1, (n_input, n_hidden))
b1 = np.zeros((1, n_hidden))
W2 = np.random.uniform(-0.1, 0.1, (n_hidden, n_output))
b2 = np.zeros((1, n_output))

#sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)

#binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#forward propagation function
def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

#backward propagation function
def backward_propagation(X, y, z1, a1, z2, a2, W1, W2):
    m = X.shape[0]

    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

batch_size = 32
learning_rate = 0.05
epochs = 150

n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

train_losses = []
test_accuracies = []

for epoch in range(epochs):
    indices = np.random.permutation(n_samples)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]

        z1, a1, z2, a2 = forward_propagation(X_batch, W1, b1, W2, b2)

        batch_loss = binary_cross_entropy(y_batch, a2)
        epoch_loss += batch_loss

        dW1, db1, dW2, db2 = backward_propagation(X_batch, y_batch, z1, a1, z2, a2, W1, W2)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    avg_train_loss = epoch_loss / n_batches
    train_losses.append(avg_train_loss)

    _, _, _, y_test_pred = forward_propagation(X_test, W1, b1, W2, b2)
    y_test_pred_class = (y_test_pred >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred_class)
    test_accuracies.append(test_accuracy)

_, _, _, y_train_final = forward_propagation(X_train, W1, b1, W2, b2)
final_train_loss = binary_cross_entropy(y_train, y_train_final)

_, _, _, y_test_final = forward_propagation(X_test, W1, b1, W2, b2)
y_test_pred_final = (y_test_final >= 0.5).astype(int)
final_test_accuracy = accuracy_score(y_test, y_test_pred_final)

print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Test Set Accuracy: {final_test_accuracy:.4f}")


#Task2

plt.figure(figsize=(14, 6))
plt.plot(range(1, epochs + 1), train_losses, 'b-', linewidth=2, label = 'Training Loss')
plt.plot(range(1, epochs + 1), test_accuracies, 'r-', linewidth=2, label ='Test Accuracy')
plt.xlabel(' Number of Epochs', fontsize=18)
plt.ylabel('Training Loss and Test Accuracy', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Training Loss and Test Accuracy over all Epochs', fontsize=18, fontweight='bold')
plt.legend(loc='best', fontsize = 18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task2_performance_curves.pdf', dpi=1000, bbox_inches='tight')
plt.show()

_, _, _, y_train_pred = forward_propagation(X_train, W1, b1, W2, b2)
y_train_pred_class = (y_train_pred >= 0.5).astype(int)
train_accuracy = accuracy_score(y_train, y_train_pred_class)

print(f"\nTask 2 - Model Evaluation:")
print(f"Final Training Accuracy: {train_accuracy:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
print(f"Accuracy Gap: {abs(train_accuracy - final_test_accuracy):.4f}")


#Task 3

print("Task 3 - Hyperparameter Investigation: Learning Rate")

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
epochs_task3 = 1000         
colors = ['blue', 'orange', 'green', 'red', 'purple']
task3_histories = {} 

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    #re-initialise identical weights for each LR
    np.random.seed(42)
    W1_lr = np.random.uniform(-0.1, 0.1, (n_input, n_hidden))
    b1_lr = np.zeros((1, n_hidden))
    W2_lr = np.random.uniform(-0.1, 0.1, (n_hidden, n_output))
    b2_lr = np.zeros((1, n_output))

    epoch_test_acc = [] #accuracy recorded after each epoch
    #mini-batch training
    for epoch in range(epochs_task3):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(n_batches):
            start = i * batch_size
            end   = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            z1, a1, z2, a2 = forward_propagation(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_propagation(
                X_batch, y_batch, z1, a1, z2, a2, W1, W2)

            W1 -= lr * dW1;  b1 -= lr * db1
            W2 -= lr * dW2;  b2 -= lr * db2
        #record test accuracy this epoch
        _, _, _, y_pred = forward_propagation(X_test, W1, b1, W2, b2)
        y_class = (y_pred >= 0.5).astype(int)
        epoch_test_acc.append(accuracy_score(y_test, y_class))

    task3_histories[lr] = epoch_test_acc
    #per-LR summary
    best_acc   = max(epoch_test_acc)
    best_epoch = epoch_test_acc.index(best_acc) + 1   #1 indexed
    final_acc  = epoch_test_acc[-1]

    print(f"  Final Test Accuracy:  {final_acc:.4f}")
    print(f"  Best Test Accuracy:   {best_acc:.4f}")
    print(f"  Best Epoch:           {best_epoch}")

#plot the learning rates 
plt.figure(figsize=(12, 6))
for idx, lr in enumerate(learning_rates):
    plt.plot(range(1, epochs_task3 + 1),
             task3_histories[lr],
             label=f'Learning Rate = {lr}',
             color=colors[idx])

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Test Accuracy', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim([0.5, 0.9])
plt.title('Test Accuracy over Epochs for Different Learning Rates', fontsize=18, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('task3_learning_rate_comparison.pdf')
plt.show()

#summary table
print("\nTask 3 - Summary of Results")
print(f"{'Learning Rate':>15} | {'Best Acc':>9} | {'Best Epoch':>10} | {'Final Acc':>9}")
print("-" * 55)
for lr in learning_rates:
    acc   = task3_histories[lr]
    best  = max(acc)
    ep    = acc.index(best) + 1
    final = acc[-1]
    print(f"{lr:>15.3f} | {best:>9.4f} | {ep:>10} | {final:>9.4f}")