import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

# Load MNIST Dataset
mnist = tf.keras.datasets.mnist
# Split Training and Test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# Convert data to 32bit floats
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Scale input between 0.0 and 1.0
x_train /= 255
x_test /= 255


# KNN classifier for k=1,3
for i in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=i)

    # train  kNN classifier
    model.fit(x_train, y_train)

    # evaluate model & update the accuracies list
    score = model.score(x_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (i, score * 100))
