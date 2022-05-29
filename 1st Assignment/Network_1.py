import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load MNIST Dataset
mnist = tf.keras.datasets.mnist
# Split Training and Test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Convert to Categorical data matrix
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Convert data to 32bit floats
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Scale input between 0.0 and 1.0
x_train /= 255
x_test /= 255

# CNN Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# Train Model
model.fit(x_train, y_train, epochs=5)

# Evaluate model
print("\n========== Evaluation =============\n")
loss, accuracy = model.evaluate(x_test, y_test)
print("accuracy on test set: ", accuracy)
print("loss on test set: ", loss)


for x in range(10):
    img = cv.imread(f"{x}.png")[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The prediction is : {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
