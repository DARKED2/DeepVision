# DeepVision
This program represents an implementation of a neural network for recognizing handwritten digits from the well-known MNIST dataset. 
The program is written in Python using the TensorFlow library for building and training the neural network. 
In the first step, the program loads the MNIST dataset, which contains 28x28 pixel images of handwritten digits from 0 to 9, as well as their corresponding labels (digit numbers). 
The data is split into training and test sets. Next, the data is preprocessed: the input images are normalized to improve the model's performance. 
Then, the architecture of the sequential neural network is defined. 
The network consists of an input layer that converts the 28x28 input images into a one-dimensional vector, two fully-connected hidden layers with 128 neurons each and a ReLU non-linear activation function, and an output fully-connected layer with 10 neurons and a softmax activation function for multi-class classification. 
After compiling the model with the Adam optimizer, a loss function for multi-class classification, and an accuracy metric, the model is trained on the training data for 3 epochs. 
The program then evaluates the performance of the trained model on the test dataset by computing the loss and accuracy values, which are printed to the screen. 
Finally, the trained model is saved to the epic_num_reader.keras file for future use. 
This program demonstrates my skills in creating and training neural networks using TensorFlow for computer vision and pattern recognition tasks. 
It can be useful for a wide range of applications requiring handwritten character or digit recognition.
