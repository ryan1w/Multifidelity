import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.activations import tanh, linear

import pandas as pd
import streamlit as st
from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)

class DeepNeuralNetwork(tf.Module):
    def __init__(self, layer_sizes, activation=tanh):
        super(DeepNeuralNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation

        init_fn = self.param_init_fn()
        self.weights = [tf.Variable(init_fn([layer_sizes[i], layer_sizes[i + 1]]), dtype=tf.float32)
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [tf.Variable(tf.zeros([layer_sizes[i + 1]]), dtype=tf.float32)
                       for i in range(len(layer_sizes) - 1)]

    def param_init_fn(self, inittializer='glorot_uniform'):
        if inittializer == 'glorot_uniform':
            init_fn = tf.initializers.GlorotUniform()
        elif inittializer == 'glorot_normal':
            init_fn = tf.initializers.GlorotNormal()
        elif inittializer == 'he_uniform':
            init_fn = tf.initializers.HeUniform()
        else:
            init_fn = tf.initializers.HeNormal()

        return init_fn

    def __call__(self, x):
        outputs = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            outputs = self.activation(tf.matmul(outputs, w) + b)
        return tf.matmul(outputs, self.weights[-1]) + self.biases[-1]

    @tf.function
    def train_step(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            pred = self(x)
            loss = tf.reduce_mean(tf.square(y - pred))
        gradients = tape.gradient(loss, self.weights + self.biases)
        optimizer.apply_gradients(zip(gradients, self.weights + self.biases))
        return loss

    def train(self, x, y, epochs=1000, learning_rate=0.001):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=epochs // 10,
            decay_rate=0.99,
            staircase=True
        )
        optimizer = tf.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.001)

        for epoch in tqdm(range(epochs), desc='Single Fidelity Training'):
            loss = self.train_step(x, y, optimizer)

            # if (epoch + 1) % 1000 == 0:
            #     current_lr = lr_schedule(optimizer.iterations.numpy()).numpy()
            #     tqdm.write(f'Epoch {epoch+1}, LR: {current_lr}, Loss: {loss.numpy()}')

    def predict(self, x):
        pred = self(x)
        return pred
    

class MultiFidelityNN(tf.Module):
    def __init__(self, input, width, depth, output, lf_model=None):
        super(MultiFidelityNN, self).__init__()

        self.lf_model = lf_model
        if lf_model is None:
            layer_sizes = [input] + [width] * depth + [output]
            self.lf_model = DeepNeuralNetwork(layer_sizes)

        nonlinear_layers = [input + output] + [width] * depth + [output]
        self.nonlinear_model = DeepNeuralNetwork(nonlinear_layers)
        linear_layers = [input + output] +  [width // 2] * (depth // 2) + [output]
        self.linear_model = DeepNeuralNetwork(linear_layers, linear)

    @tf.function
    def train_step(self, x_lf, y_lf, x_hf, y_hf, optimizer):
        with tf.GradientTape() as tape:
            pred_lf = self.lf_model(x_lf)
            pred_hf = self(x_hf)
            loss_lf = tf.reduce_mean(tf.square(y_lf - pred_lf))
            loss_hf = tf.reduce_mean(tf.square(y_hf - pred_hf))
            loss_l2 = 1 * tf.add_n([tf.nn.l2_loss(v) for v in self.linear_model.weights])
            loss = loss_hf + loss_l2
        variables = self.nonlinear_model.weights + self.nonlinear_model.biases + self.linear_model.weights + self.linear_model.biases #+ self.lf_model.weights + self.lf_model.biases
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss, loss_lf, loss_hf, loss_l2

    def train(self, x_lf, y_lf, x_hf, y_hf, epochs=1000, learning_rate=0.001):
        optimizer = tf.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001)
        loss_history = []
        for epoch in tqdm(range(epochs), desc='Multi-Fidelity Training'):
            loss, loss_lf, loss_hf, loss_l2 = self.train_step(x_lf, y_lf, x_hf, y_hf, optimizer)
            loss_history.append(loss.numpy())

            # if epoch % 100 == 0:
            #     tqdm.write(f'Epoch {epoch}, Loss: {loss.numpy()}, loss_lf: {loss_lf.numpy()}, loss_hf: {loss_hf.numpy()}, loss_l2: {loss_l2.numpy()}')
        return loss_history

    def __call__(self, x_hf):
        pred_lf = self.lf_model(x_hf)
        x_combined = tf.concat([x_hf, pred_lf], axis=-1)
        y = self.nonlinear_model(x_combined) + self.linear_model(x_combined)
        return y
    
    def predict(self, x):
        pred = self(x)
        return pred
    
def low_fidelity_data(x):
    y = np.sin(8 * np.pi * x)
    # y = 0.5*(6*x - 2)**2*np.sin(12*x - 4) + 10*(x - 0.5) - 5
    return y

def high_fidelity_data(x):
    y = (x - np.sqrt(2.0)) * low_fidelity_data(x)**2 
    # y = (6*x - 2)**2*np.sin(12*x - 4)
    return y

def create_data_csv():
    x_lf = np.linspace(0, 1, 100).reshape(-1, 1)
    y_lf = low_fidelity_data(x_lf)
    x_hf = np.linspace(0, 1, 14).reshape(-1, 1)
    y_hf = high_fidelity_data(x_hf)
    df_lf = pd.DataFrame(np.hstack((x_lf, y_lf)), columns=["x_lf", "y_lf"])
    df_hf = pd.DataFrame(np.hstack((x_hf, y_hf)), columns=["x_hf", "y_hf"])
    df = pd.concat([df_lf, df_hf], axis=1)
    df.to_csv("data.csv", index=False)


def multi_fidelity_app():
    st.title('Multi-fidelity neural network visualization')

    file = st.file_uploader('Upload training data CSV', type=['csv'])
    if file is None:
        st.info("Waiting for CSV file upload...")
        st.stop()

    data = pd.read_csv(file)
    st.write('Preview of uploaded data:', data.head())
    x_lf = data['x_lf'].values.reshape(-1, 1).astype(np.float32)
    y_lf = data['y_lf'].values.reshape(-1, 1).astype(np.float32)
    x_hf = data['x_hf'].dropna().values.reshape(-1, 1).astype(np.float32)
    y_hf = data['y_hf'].dropna().values.reshape(-1, 1).astype(np.float32)

    x_lf = tf.convert_to_tensor(x_lf, dtype=tf.float32)
    y_lf = tf.convert_to_tensor(y_lf, dtype=tf.float32)
    x_hf = tf.convert_to_tensor(x_hf, dtype=tf.float32)
    y_hf = tf.convert_to_tensor(y_hf, dtype=tf.float32)

    x_min = tf.reduce_min(x_lf)
    x_max = tf.reduce_max(x_lf)
    x_test = tf.linspace(x_min, x_max, 500)[:, None]

    st.sidebar.header("Model Configuration")
    layer_num = st.sidebar.number_input("Number of hidden layers", min_value=1, max_value=10, value=2)
    layer_size = st.sidebar.number_input("Number of units per layer", min_value=1, max_value=100, value=20)
    epochs = st.sidebar.number_input("Epochs", min_value=1, value=2000)
    lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, format="%.4f")

    if st.button("▶️ Train Model"):
        st.write("Training started...")

        # Training low fidelity model
        layer_input = x_lf.shape[1]
        layer_output = y_hf.shape[1]
        layer_sizes = [layer_input] +  [layer_size] * layer_num + [layer_output]
        model_lf = DeepNeuralNetwork(layer_sizes)
        st.write("Training low-fidelity model...")
        model_lf.train(x_lf, y_lf, epochs=epochs, learning_rate=lr)
        y_pred_lf = model_lf.predict(x_test)
        data = pd.DataFrame({'x': x_test.numpy().flatten(), 'y': y_pred_lf.numpy().flatten()})
        st.line_chart(data.set_index('x'))

        st.write("Training high-fidelity model...")
        model_hf = DeepNeuralNetwork(layer_sizes)
        model_hf.train(x_hf, y_hf, epochs=epochs, learning_rate=lr)
        y_pred_hf = model_hf.predict(x_test)
        data_hf = pd.DataFrame({'x': x_test.numpy().flatten(), 'y': y_pred_hf.numpy().flatten()})
        st.line_chart(data_hf.set_index('x'))

        st.write("Training multi-fidelity model...")
        model_mf = MultiFidelityNN(layer_input, layer_size, layer_num, layer_output, lf_model=model_lf)
        model_mf.train(x_lf, y_lf, x_hf, y_hf, epochs=epochs, learning_rate=lr)
        predictions = model_mf.predict(x_test)
        data_mf = pd.DataFrame({'x': x_test.numpy().flatten(), 'y': predictions.numpy().flatten()})
        st.line_chart(data_mf.set_index('x'))

        st.success("✅ Training completed!")


if __name__ == '__main__':
    # create_data_csv()
    multi_fidelity_app()