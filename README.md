# Sentimental-Analysis
Sentiment Analysis with Using RNN

In this project, it will be tried to predict whether a situation given to the model is positive or negative. In the project, we used the NumPy, Pandas and Keras libraries for the RNN model.

All the data used in model training were collected on Hepsiburada.com. 250 thousand comments and evaluation scores for these comments were collected. When tagging the data, 1- and 2-star reviews are negative, 4- and 5-star reviews are positive, and 3-star reviews are neutral, so they are not included in the data. we divided this dataset and allocated 80% for training and 20% for testing.

The tagged comments were first tokenized, and a vocabulary was created. Then, by creating an Embedding layer, a vector representing that word was created for each word.

During the training, the embedding size was set to 50, so we will create an overall 50-length vector for each word. Using the CuDNNGRU RNN model, we will train the model on the comments on the dataset, then the RNN network will be connected to a neural network with a single neuron, according to the result from this network, it will predict whether the comment given to the model is positive or negative. Since we have only two classes, we use the binary cross entropy loss function. Since the Sigmoid function squeezes the data between 0 and 1, if the result is close to 1, the comment will predict negative if the comment is close to positive 0.