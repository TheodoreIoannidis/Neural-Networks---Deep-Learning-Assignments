from tensorflow.keras import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# Load MNIST Dataset
# Split Training and Test sets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
pca = PCA(0.95)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

y_test = y_test % 2
y_train = y_train % 2
# Convert data to 32bit floats
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Scale input between 0.0 and 1.0
x_train /= 255
x_test /= 255

clf = svm.SVC(kernel="rbf", gamma=10, C=100)

t0 = time.time()
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
t1 = time.time() - t0
print("Time elapsed: ", t1)
print("classification report: \n", metrics.classification_report(y_test, prediction))
print("accuracy: ", metrics.accuracy_score(y_test, prediction))
