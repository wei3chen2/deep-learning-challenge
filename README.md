# Deep-Learning-Challenge
![machinelearning.tmp](machinelearning.tmp)
## Overview
This project is aimed to build a model that can  help  Alphabet Soup, a nonprofit foundation, to choose the most promising applicants for charity funding. You will use machine learning and neural networks to design a binary classificaier that can predict whether appliacants will be successful or not, based on a CSV dataset of more than 34,000 organizations that have received funding from Alphabet Soup during the past several years.



## Project outline

### Step 1: Preprocess and explore the Dataset using Pandas and scikit-learn
   * Read in the charity_data.csv to a Pandas DataFrame
   * Pick a cutoff point to bin "rare" categorical variables
   * Split the data into training and testing datasets.
   * Scale the training and testing features datasets by creating a StandardScaler instance
### Step 2: Compile, Train, and Evaluate the neural network Model using TensorFlow and Keras
   * Continue using the file in Google Colab in which you performed the preprocessing steps
   * Create a neural network model by assigning the number of input features and nodes for each     layer using TensorFlow and Keras
   * Create the several hidden layers and an output layer with an appropriate activation function.
   * Compile and train the model.
   * Evaluate the model using the test data to determine the loss and accuracy.
   * Save and export the results to an HDF5 file.
 ### Step 3: Optimize the Model
   After adjusting the input data and optimization, the predictive accuracy has outperformed the target 75% and reached 80.005%. The following methods are used: 
   * Dropping fewer columns.
   * Creating more bins for rare occurrences in columns.
   * Increasing or decreasing the number of values for each bin.
   * Add more neurons to a hidden layer.
   * Add more hidden layers.
   * Use different activation functions for the hidden layers.
   * Add or the number of epochs to the training regimen.
 ### Step 4: Write a Report on the Neural Network Model 
