# NEURAL NETWROKS- SECOND ASSIGNMENT:

In this assignment I had to compare the performances of different models on a multiclass classification problem.
I built a Support Vector Classifier, two kNN models with 1 and 3 nearest neighbors respectively and finally a model that uses the Nearest Centroid algorithm.

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

This time, the problem we have to solve is to classify the handwritten digits from the MNIST dataset, as ODD or EVEN numbers.

To do this, we have to set new labels, like so:

    # Labels from (0,1,...,9) to (0,1)
    y_train = y_train % 2
    y_test = y_test % 2

Now the odd numbers have labels equal to 1, since (odd_number)mod2= 1 and even numbers have labels equal to 0, since (even_number)mod2= 0.

# Support Vector Classifier.

I noticed that the training time for the SVCs was really slow. Because I wanted to perform multiple runs for different values of each parameter, i decidedd to do PCA on the data and keep 95% of the information of the data. There was substantial improvement in speed.

After trying out different kernels (rbf, linear etc.) and different values for gamma and C, I found that the best accuracy achieved by the SVM was 98,3% for:
    
     kernel= 'rbf', gamma=10, C=100. 

Time elapsed (for fitting and predicting): 463 seconds 

Using kernel = 'linear' produced worse results but the models were faster.
With that being said, the worse results out of the models I tried were: 

    kernel= 'rbf', gamma=0.1, C= 0.1
    accuracy = 62.64 % 
    time elapsed = 1180 seconds

# K- Nearest Neighbors.

Next, I built a KNN classifier using scikit-learn, for the same classification problem. 

I tried k= 1,..., 10 and all the models achieved an accuracy higher than 98%.

The highest accuracy was 98.65 for k=6. Also with k = 4 the model had 98.64%

The lowest accuracy was 98.31% for k =9.

It is worth noting that the kNN models were a lot faster than the SVCs.

# Nearest Centroid.

Finally, I used scikit-learn's NearestCentroid and fit the data to the model.
The results were the lowest out of all the other models with 80.26% accuracy.

Once again, it is worth noting that although the accuracy of the model is almost 20% lower than that of the others, it is by far the fastest model with 0.15 seconds for fitting and predictiions.

# Conclusion.

In conclusion, kNN was the second fastest and achieved the highest accuracy. The Support Vector Classifier achieved the second highest accuracy but was a lot slower even with PCA applied to the data. The Nearest Centroid model had the worst accuracy but was the fastest out of the three.

