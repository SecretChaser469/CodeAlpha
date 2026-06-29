# ==========================================
# TASK 3: HANDWRITTEN CHARACTER RECOGNITION
# ==========================================
!pip install -q numpy tensorflow matplotlib

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load and Preprocess MNIST Data
print("Fetching MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape for CNN (using a subset for faster training demonstration)
X_train = X_train[:5000].reshape(-1, 28, 28, 1) / 255.0
y_train = y_train[:5000]
X_test = X_test[:1000].reshape(-1, 28, 28, 1) / 255.0
y_test = y_test[:1000]

# 2. Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Train Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

# 4. Evaluate & Visualize
_, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n--- TASK 3 EVALUATION REPORT ---")
print(f"CNN Character Recognition Accuracy: {acc*100:.2f}%")

predictions = model.predict(X_test[:5], verbose=0)
plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.suptitle('Handwritten Digit Predictions')
plt.show()
