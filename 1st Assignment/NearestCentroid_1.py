import tensorflow as tf
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

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

model = NearestCentroid(metric="euclidean")
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("\n accuracy:", accuracy_score(y_test, prediction))

"""
for i in range(0,10,1):
  #if (y_test[i]) != model.predict(x_test[i,:].reshape(1,-1)):
   print('digit ',y_test[i],' was classified as: ' ,model.predict(x_test[i,:].reshape(1,-1)))  
"""
