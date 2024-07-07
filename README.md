# Traffic-Sign-Recognition
Traffic Sign Recognition Using Convolutional Neural Networks and Deeplake
## File Descriptions

### `data_preprocessing.py`

This script is responsible for loading, preprocessing, and saving the GTSRB dataset. It performs the following tasks:

- **Load the Dataset**: Loads the training and testing datasets from Deeplake.
- **Process Images**: Converts images from RGB to grayscale and resizes them to a consistent size (32x32 pixels).
- **Process Labels**: Converts labels to one-hot encoded vectors.
- **Save Data**: Saves the preprocessed images and labels into compressed NumPy files for easy loading during training.

### `model.py`

This script defines the Convolutional Neural Network (CNN) model used for traffic sign recognition. It uses TensorFlow and Keras to build a sequential model with the following layers:

- **Convolutional Layers**: Extracts features from the input images.
- **MaxPooling Layers**: Reduces the spatial dimensions of the feature maps.
- **Dropout Layers**: Prevents overfitting by randomly setting a fraction of input units to zero.
- **Fully Connected Layers**: Performs classification based on the extracted features.
- **Output Layer**: Produces the final classification output with softmax activation.

### `train.py`

This script is responsible for training and evaluating the CNN model. It performs the following tasks:

- **Load Data**: Loads the preprocessed training and testing data.
- **Create Model**: Builds the CNN model using the `create_model` function from `model.py`.
- **Train Model**: Trains the model on the training data and validates it on the testing data.
- **Evaluate Model**: Evaluates the model's performance on the testing data and prints the accuracy and loss.
- **Generate Reports**: Produces a classification report and confusion matrix to analyze the model's performance.

### `requirements.txt`

This file lists all the dependencies required for the project. You can install them using pip:

```text
deeplake
numpy
scikit-image
scikit-learn
matplotlib
tensorflow
