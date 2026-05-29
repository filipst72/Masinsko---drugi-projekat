import torch as th
import numpy as np

# ==========================================
# 1. Load and Preprocess Recorded Data
# ==========================================
x = np.load('observations.npy', allow_pickle=True)
y = np.load('actions.npy', allow_pickle=True)

x = np.stack(x).reshape(len(x), -1)


x_tensor = th.tensor(x, dtype=th.float32)
y_tensor = th.tensor(y, dtype=th.long)


class NN:
    def __init__(self, input_size=108, hidden_size=128, output_size=7, device='cpu'):
        self.device = device
        
        self.W1 = th.randn(input_size, hidden_size, device=device) * np.sqrt(2.0 / input_size)
        self.W2 = th.randn(hidden_size, output_size, device=device) * np.sqrt(2.0 / hidden_size)
        self.b1 = th.zeros(hidden_size, device=device)
        self.b2 = th.zeros(output_size, device=device)

        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def relu(self, x):
        return th.maximum(th.tensor(0.0, device=self.device), x)

    def relu_izvod(self, x):
        return (x > 0).float()

    def softmax(self, x):
        exp_x = th.exp(x - th.max(x, dim=1, keepdim=True)[0])
        return exp_x / exp_x.sum(dim=1, keepdim=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def __call__(self, X):
        return self.forward(X)

    def backward(self, X, y_true, y_pred):
        batch_size = X.shape[0]
        dz2 = y_pred - y_true  
        self.dW2 = (self.a1.T @ dz2) / batch_size
        self.db2 = dz2.sum(dim=0) / batch_size
        
        dz1 = (dz2 @ self.W2.T) * self.relu_izvod(self.z1)
        self.dW1 = (X.T @ dz1) / batch_size
        self.db1 = dz1.sum(dim=0) / batch_size

    def update_weights(self, learning_rate):
        self.W1 = self.W1 - learning_rate * self.dW1
        self.b1 = self.b1 - learning_rate * self.db1
        self.W2 = self.W2 - learning_rate * self.dW2
        self.b2 = self.b2 - learning_rate * self.db2


model = NN(input_size=108, hidden_size=128, output_size=7)

epochs = 1000
batch_size = 32
learning_rate = 0.01
num_samples = x_tensor.shape[0]

print(f"Starting training on with {num_samples} data points...")

for epoch in range(epochs):
    permutation = th.randperm(num_samples)
    epoch_loss = 0.0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_tensor[indices], y_tensor[indices]

        y_pred = model(batch_x)

        y_true_one_hot = th.nn.functional.one_hot(batch_y, num_classes=7).float()
        loss = -th.sum(y_true_one_hot * th.log(y_pred + 1e-8)) / batch_x.shape[0]
        epoch_loss += loss.item()
        num_batches += 1

        model.backward(batch_x, y_true_one_hot, y_pred)
        model.update_weights(learning_rate)
        
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {epoch_loss / num_batches:.4f}")


th.save({
    'W1': model.W1,
    'W2': model.W2,
    'b1': model.b1,
    'b2': model.b2
}, 'bc_model.pth')

print("Training finished! Model parameters successfully saved to 'bc_model.pth'.")