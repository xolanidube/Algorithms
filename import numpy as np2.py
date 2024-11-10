import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    PReLU,
    BatchNormalization,
)
from tensorflow.keras.models import Model

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images to include channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)




# Input layer
input_layer = Input(shape=(28, 28, 1))

# Level 1 - Basic features
x = Conv2D(32, (3, 3), padding='same')(input_layer)
x = BatchNormalization()(x)
x = PReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Level 2 - Intermediate features
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Level 3 - High-level features
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Flatten and fully connected layers
x = Flatten()(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = PReLU()(x)

# Output layer
output_layer = Dense(10, activation='softmax')(x)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Print the model summary
model.summary()


# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test)
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.show()


# Make predictions on test set
predictions = model.predict(x_test)

# Plot some test images with predicted and true labels
num_images = 5
plt.figure(figsize=(15, 3))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(
        f"Predicted: {np.argmax(predictions[i])}\nTrue: {np.argmax(y_test[i])}"
    )
    plt.axis('off')
plt.tight_layout()
plt.show()
