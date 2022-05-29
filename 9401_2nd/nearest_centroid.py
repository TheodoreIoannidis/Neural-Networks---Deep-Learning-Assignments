from tensorflow.keras import datasets
from sklearn.neighbors import NearestCentroid
from sklearn import metrics
import time

# Load MNIST Dataset
# Split Training and Test sets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Reshape data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Convert data to 32bit floats
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Scale input between 0.0 and 1.0
x_train /= 255
x_test /= 255
# Labels from (0,...,9) to (0,1)
y_train = y_train % 2
y_test = y_test % 2

model = NearestCentroid(metric="euclidean")
t0 = time.time()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
t1 = time.time() - t0
print("Time elapsed: ", t1)
print("\n accuracy: ", metrics.accuracy_score(y_test, prediction))
