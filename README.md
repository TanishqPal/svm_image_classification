# TensorFlow MNIST Classifier

This repository contains a simple TensorFlow-based neural network model for classifying MNIST handwritten digits dataset. The model is implemented using TensorFlow 2.9.2.

## Setup

Before running the code, make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install tensorflow
```

Usage
Import the TensorFlow library and check the version:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
```

Expected output:
TensorFlow version: 2.9.2

Load and preprocess the MNIST dataset:

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

Define the neural network model:

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```
Make predictions and apply softmax:

```python
predictions = model(x_train[:1]).numpy()
softmax_predictions = tf.nn.softmax(predictions).numpy()
```
Define the loss function:

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```
Calculate the loss for a sample:

```python
loss = loss_fn(y_train[:1], predictions).numpy()
```
Compile the model:

```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```
Train the model:

```python
model.fit(x_train, y_train, epochs=5)
```
Evaluate the model on the test set:

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
```
Create a probability model:

```python
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```
Make predictions using the probability model:

```python
prediction_probabilities = probability_model(x_test[:5])
```

License
This project is licensed under the MIT License - see the LICENSE file for details.
Feel free to modify the content according to your needs, and make sure to update any necessary links or additional information.
