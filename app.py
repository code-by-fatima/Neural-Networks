
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import make_circles, make_moons
import warnings
warnings.filterwarnings('ignore')

class ActivationFunctions:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    @staticmethod
    def sigmoid_derivative(z):
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z)**2
    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        return np.where(z > 0, 1, alpha)

class LossFunctions:
    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_history = []
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        for epoch in range(self.epochs):
            errors = 0
            for i in range(n_samples):
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if z >= 0.0 else 0
                error = y[i] - y_pred
                errors += int(error != 0)
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error
            self.errors_history.append(errors)
        return self
    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        return np.where(output >= 0.0, 1, 0)

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, activation='relu', optimizer='adam'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.L = len(layer_sizes)
        self.weights = []
        self.biases = []
        self.gradient_history = []
        self.loss_history = []
        self.val_loss_history = []
        self._init_weights()
        self.opt_name = optimizer
        self.opt_m = [{} for _ in range(self.L - 1)]
        self.opt_v = [{} for _ in range(self.L - 1)]
        self.opt_t = 0

    def _init_weights(self):
        for i in range(self.L - 1):
            std = np.sqrt(2.0 / self.layer_sizes[i]) if self.activation == 'relu' else np.sqrt(1.0 / self.layer_sizes[i])
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * std)
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))

    def _activate(self, Z, deriv=False):
        if self.activation == 'relu':
            return (Z > 0).astype(float) if deriv else np.maximum(0, Z)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z)**2 if deriv else np.tanh(Z)
        elif self.activation == 'leaky_relu':
            return np.where(Z > 0, 1, 0.01) if deriv else np.where(Z > 0, Z, 0.01*Z)
        else:
            s = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            return s * (1 - s) if deriv else s

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    def _sigmoid_deriv(self, Z):
        s = self._sigmoid(Z)
        return s * (1 - s)

    def _opt_update(self, w, dw, layer_idx):
        lr = self.learning_rate
        if self.opt_name == 'sgd':
            return w - lr * dw
        elif self.opt_name == 'momentum':
            beta = 0.9
            if 'v' not in self.opt_m[layer_idx]:
                self.opt_m[layer_idx]['v'] = np.zeros_like(w)
            self.opt_m[layer_idx]['v'] = beta * self.opt_m[layer_idx]['v'] - lr * dw
            return w + self.opt_m[layer_idx]['v']
        else:
            beta1, beta2, eps = 0.9, 0.999, 1e-7
            self.opt_t += 1
            if 'm' not in self.opt_m[layer_idx]:
                self.opt_m[layer_idx]['m'] = np.zeros_like(w)
                self.opt_v[layer_idx]['v'] = np.zeros_like(w)
            self.opt_m[layer_idx]['m'] = beta1 * self.opt_m[layer_idx]['m'] + (1 - beta1) * dw
            self.opt_v[layer_idx]['v'] = beta2 * self.opt_v[layer_idx]['v'] + (1 - beta2) * (dw**2)
            m_hat = self.opt_m[layer_idx]['m'] / (1 - beta1**self.opt_t)
            v_hat = self.opt_v[layer_idx]['v'] / (1 - beta2**self.opt_t)
            return w - lr * m_hat / (np.sqrt(v_hat) + eps)

    def forward(self, X):
        self.A = [X]
        self.Z = []
        A = X
        for i in range(self.L - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.Z.append(Z)
            A = self._activate(Z) if i < self.L - 2 else self._sigmoid(Z)
            self.A.append(A)
        return A

    def compute_loss(self, y_true, y_pred, loss_type='binary_crossentropy', lambda_reg=0.0):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = np.mean((y_true - y_pred)**2) if loss_type == 'mse' else -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        if lambda_reg > 0:
            loss += (lambda_reg / (2 * m)) * sum(np.sum(w**2) for w in self.weights)
        return loss

    def backward(self, y_true, lambda_reg=0.0):
        m = y_true.shape[0]
        dA = -(y_true / (self.A[-1] + 1e-8) - (1 - y_true) / (1 - self.A[-1] + 1e-8))
        grads_w, grads_b = [], []
        max_grad = 0
        for i in range(self.L - 2, -1, -1):
            dZ = dA * self._sigmoid_deriv(self.Z[i]) if i == self.L - 2 else dA * self._activate(self.Z[i], deriv=True)
            dW = np.dot(self.A[i].T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            if lambda_reg > 0:
                dW += (lambda_reg / m) * self.weights[i]
            grads_w.insert(0, dW)
            grads_b.insert(0, dB)
            max_grad = max(max_grad, np.max(np.abs(dZ)))
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
        self.gradient_history.append(max_grad)
        return grads_w, grads_b

    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2,
            loss_type='binary_crossentropy', lambda_reg=0.0):
        n = X.shape[0]
        val_size = int(n * validation_split)
        if val_size == 0:
            X_train, X_val, y_train, y_val = X, X, y, y
        else:
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
        self.loss_history = []
        self.val_loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = max(1, len(X_train) // batch_size)
            for i in range(n_batches):
                Xb = X_train[i*batch_size:(i+1)*batch_size]
                yb = y_train[i*batch_size:(i+1)*batch_size]
                if len(Xb) == 0:
                    continue
                out = self.forward(Xb)
                epoch_loss += self.compute_loss(yb, out, loss_type, lambda_reg)
                gw, gb = self.backward(yb, lambda_reg)
                for j in range(len(self.weights)):
                    self.weights[j] = self._opt_update(self.weights[j], gw[j], j)
                    self.biases[j] -= self.learning_rate * gb[j]
            val_out = self.forward(X_val)
            self.loss_history.append(epoch_loss / n_batches)
            self.val_loss_history.append(self.compute_loss(y_val, val_out, loss_type, lambda_reg))
        return self.loss_history, self.val_loss_history

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def predict_proba(self, X):
        return self.forward(X)


st.set_page_config(page_title="Neural Networks ", layout="wide")
st.title("🧠 Neural Network Visualizer")
st.markdown(" Interactive Demo")

st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Architecture")
h1 = st.sidebar.slider("Hidden Layer 1 neurons", 2, 32, 4)
h2 = st.sidebar.slider("Hidden Layer 2 neurons", 2, 32, 4)
layer_sizes = [2, h1, h2, 1]
params = sum(layer_sizes[i]*layer_sizes[i+1] + layer_sizes[i+1] for i in range(len(layer_sizes)-1))
st.sidebar.write(f"Network: {layer_sizes} | Params: {params}")

st.sidebar.subheader(" Activation")
activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "leaky_relu"])

st.sidebar.subheader("Loss Function")
loss_type = st.sidebar.selectbox("Loss Function", ["binary_crossentropy", "mse"])

st.sidebar.subheader("Learning Rate")
lr = st.sidebar.slider("Learning Rate", 0.0001, 0.5, 0.01, step=0.001)

st.sidebar.subheader("Epochs")
epochs = st.sidebar.slider("Epochs", 10, 500, 100)

st.sidebar.subheader("Batch Size")
batch_size = st.sidebar.slider("Batch Size (1=SGD, 32=Mini-batch)", 1, 128, 32)

st.sidebar.subheader("Optimizer")
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "momentum"])

st.sidebar.subheader("Regularization")
lambda_reg = st.sidebar.slider("L2 Lambda", 0.0, 0.1, 0.0, step=0.001)


st.header("Dataset Selection")
dataset = st.radio("Choose Dataset", ["XOR Problem", "Circle Classification", "Two Moons"], horizontal=True)

if dataset == "XOR Problem":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    st.info("Perceptron FAILS here | MLP SOLVES this!")
elif dataset == "Circle Classification":
    X, yf = make_circles(n_samples=300, noise=0.1, random_state=42)
    y = yf.reshape(-1,1).astype(float)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    st.info("Traditional ML (logistic regression) fails here, NN succeeds!")
else:
    X, yf = make_moons(n_samples=300, noise=0.1, random_state=42)
    y = yf.reshape(-1,1).astype(float)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    st.info("Try increasing L2 lambda to see regularization effect!")

st.header("Perceptron Demo")
if st.checkbox("▶ Show Perceptron failing on XOR"):
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_xor = np.array([0,1,1,0])
    p = Perceptron(epochs=100)
    p.fit(X_xor, y_xor)
    pred = p.predict(X_xor)
    acc = np.mean(pred == y_xor)
    st.error(f" Perceptron on XOR: {acc:.0%} — FAILS (can only draw straight line)")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(p.errors_history, color='red', linewidth=2)
    ax.set_title("Perceptron errors never reach 0 on XOR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Errors")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

# TRAIN
st.header("Train Network (Topics 6, 9, 10, 11, 12, 13, 14, 15, 16, 17)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Architecture", str(layer_sizes))
c2.metric("Activation", activation.upper())
c3.metric("Optimizer", optimizer.upper())
c4.metric("Parameters", params)

if st.button("▶️ TRAIN NOW", use_container_width=True):
    with st.spinner("Training in progress..."):
        network = MLP(layer_sizes, learning_rate=lr, activation=activation, optimizer=optimizer)
        bs = min(batch_size, len(X))
        vs = 0 if len(X) < 10 else 0.2
        train_losses, val_losses = network.fit(
            X, y, epochs=epochs, batch_size=bs,
            validation_split=vs, loss_type=loss_type, lambda_reg=lambda_reg
        )

    st.success("Training Complete!")

    predictions = network.predict(X)
    accuracy = np.mean(predictions == y)

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{accuracy:.2%}")
    m2.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
    m3.metric("Final Val Loss", f"{val_losses[-1]:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Topics 9,11,13: Loss Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, name='Train Loss', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(y=val_losses, name='Val Loss', line=dict(color='red', width=2, dash='dash')))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Topic 16: Gradient Flow")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=network.gradient_history, name='Max Gradient', line=dict(color='green', width=2)))
        fig2.update_layout(xaxis_title="Epoch", yaxis_title="Gradient Magnitude", height=350)
        st.plotly_chart(fig2, use_container_width=True)
        avg_g = np.mean(network.gradient_history[-10:]) if network.gradient_history else 0
        if avg_g > 0.01:
            st.success(f"Gradients stable: {avg_g:.4f}")
        else:
            st.warning(f"Gradients vanishing: {avg_g:.6f} → Switch to ReLU!")

    if len(X) > 4:
        st.subheader("Topic 6: Decision Boundary (Forward Propagation Output)")
        h_step = 0.02
        x0_min, x0_max = X[:,0].min()-0.1, X[:,0].max()+0.1
        x1_min, x1_max = X[:,1].min()-0.1, X[:,1].max()+0.1
        xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h_step),
                             np.arange(x1_min, x1_max, h_step))
        Z = network.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig3, ax = plt.subplots(figsize=(8, 5))
        ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
        ax.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='RdBu', edgecolors='black', s=50)
        ax.set_title(f"Topic 6: Decision Boundary | Accuracy: {accuracy:.2%}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        st.pyplot(fig3)
        plt.close()

    st.subheader("All Topics Summary")
    iter_per_epoch = max(1, len(X) // batch_size)
    st.markdown(f"""
| Concept | Your Config |
|---------|-------------|
| Traditional ML Limitations | Dataset shows NN advantage |
| Bio vs Artificial Neuron | Each node = artificial neuron |
| Perceptron | Fails on XOR (demo above) |
| MLP | {layer_sizes} solves non-linear |
|Architecture | {layer_sizes} = {params} parameters |
|Forward Propagation | Decision boundary above |
|Activation Function | **{activation.upper()}** |
|Loss Function | **{loss_type}** |
|Gradient Descent | Mini-batch with batch={batch_size} |
|Backpropagation | Chain rule → gradient plot |
|Learning Rate | **{lr}** |
|Batch Size | **{batch_size}** ({'SGD' if batch_size==1 else 'Mini-batch' if batch_size < len(X) else 'Full Batch'}) |
|Epochs vs Iterations | {epochs} epochs × {iter_per_epoch} iters = **{epochs*iter_per_epoch} total** |
|Optimizer | **{optimizer.upper()}** |
|Weight Init | **{'He (ReLU)' if activation == 'relu' else 'Xavier'}** |
|Vanishing Gradients | {' Stable' if avg_g > 0.01 else ' Vanishing'} ({avg_g:.2e}) |
|Regularization | L2 λ = **{lambda_reg}** |
    """)

st.markdown("---")
st.caption("Neural Networks | Practical Internship Project")
