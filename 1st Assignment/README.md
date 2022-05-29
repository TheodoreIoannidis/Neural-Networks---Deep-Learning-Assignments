# NEURAL NETWROKS- FIRST ASSIGNMENT:

In this assignment I had to compare the performances of different models on a multiclass classification problem.
I built a MLP training with back-propagation, two kNN models with 1 and 3 nearest neighbors and finally a model that uses the Nearest Centroid algorithm.

## Data
The data I used to train and evaluate these models is MNIST, which is  a database of handwritten digits (0-9). 
It has a training set of 60,000 examples, and a test set of 10,000 examples. The pictures in th dataset are 28x28 pixels with grayscale values (in the range of 0-255). 


### Load Data    

    # Load MNIST Dataset
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    # Split Training and Test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


With the above code we can load the data and split it into train and test sets. I used the default split (60k samples for training, 10k samples for testing).


# Neural Network

The data needs a little preprocessing. First, I converted the labels from integers, into binary class vectors, so that they match the formatting of the NN's output layer.
Then, I converted the grayscale values of the pixels into 32 bit floats and also I scaled them from 0-255 to 0-1, by dividing with 255. 
   
    # Convert to Categorical data matrix
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Convert data to 32bit floats
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Scale input between 0.0 and 1.0
    x_train /= 255
    x_test /= 255


The Neural Network I built has an input layer which consists of 784 neurons (as many as the pixels of each image 28*28= 784), a hidden layer with 128 neurons with their activation function set to ReLu and an output layer with 10 neurons (as many as the number of classes ( 0-9 ) ) with Softmax set as their activation function.

After trying different values for the training parameters, I found that the best results were produced using the Adam optimizer, categorical crossentropy loss function and training for 5 epochs. 

    Train Set: 
        accuracy = 0.9862
        loss = 0.0446
    Test Set:
        accuracy = 0.978
        loss = 0.0746
    
I also tried adding another hidden layer with 128 neurons (Relu activation function) everything else stayed the same (optimizer, loss function etc.).
The accuracy improved slightly both in training and in testing. I added another layer with 64 neurons, but the model did not improve further. 

I also tried other activation functions (such as sigmoid and tanh) in the hidden layers, and the performance was substantially worse in every case.

## Custom Handwritten digits.

I wanted to test the model on pictures that did not go through the same kind of collection precess as the digits from MNIST. For that purpose, I drew 10 digits in MS Paint, saved them as x.png ( where x= 0, 1,..., 9 ). I to set the dimensions of each image to 28x28 pixels, i used the pencil and drew digit x on a white background. I processed the images with a few lines of code in order to make them as close to the training samples as possible (invert black to white and vice versa).

I used the cv2 library in order to read the images, processed them and fed them into the trained model. The nn predicted digits 0,1,2,3,5,6,7 correctly and failed with digits 4,8,9 which were classified as 9,3,3 respectively.  

# K-Nearest Neighbors

I used scikit-learn's kNeighborsClassifier and tested the performance for different values of k (nearest neighbors).

The preprocessing needed:

    # Reshape data
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Convert data to 32bit floats
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Scale input between 0.0 and 1.0
    x_train /= 255
    x_test /= 255

The assignment asked to compare the results for k=1 and 3, but I wanted find out how the accuracy changes for a bigger set of different values of k.
I created 10 models and compared their performance on the test set. The results were:

    k=1, accuracy= 96.91%
    k=2, accuracy= 96.27%
    k=3, accuracy= 97.05%
    k=4, accuracy= 96.82%
    k=5, accuracy= 96.88%
    k=6, accuracy= 96.77%
    k=7, accuracy= 96.94%
    k=8, accuracy= 96.70%
    k=9, accuracy= 96.59%
    k=10, accuracy= 96.65%

We get the best result for k = 3. In every case, the accuracy is lower than the one achieved by the neural network. 

# Nearest Centroid.

Finally, we were asked to compare the abovce results with the performance of the Nearest Centroid algorithm.

The preprocessing needed is the same as above. Reshape, Convert to 32-bit floats and scale from 0-255 to 0-1.

I used NearestCentroid from the scikit-learn library in order to create the model. The accuracy that the model achieves is substantially lower than the ones achieved by the other models.

accuracy= 0.8203

# Conclusion.

The biggest accuracy is achieved by the neural network (97,8%).
The K- Nearest neighbors Model with k=3 achieved the second highest accuracy (97.05%), and the nearest centroid model had the worst performance by far (82.03%).


See:    
    
    http://yann.lecun.com/exdb/mnist/

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

    https://www.tensorflow.org/api_docs/python/tf/keras

    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist