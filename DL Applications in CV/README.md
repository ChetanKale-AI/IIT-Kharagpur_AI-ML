# AlexNet Implementation for MNIST Classification  

This repository contains a Jupyter notebook demonstrating various image processing techniques and building a trimmed version of the AlexNet architecture for classifying the MNIST dataset.

# Introduction

This project explores:  
1. Basic operations on digital images using libraries like skimage and matplotlib.
2. Implementation of AlexNet, a convolutional neural network (CNN) designed for image classification, adapted for the MNIST dataset.
3. Training, testing, and performance evaluation of the AlexNet model using PyTorch.

# Dataset

The notebook uses the MNIST dataset, which contains grayscale images of handwritten digits (0-9). This dataset is automatically downloaded using torchvision.datasets.MNIST and is split into training and testing datasets.

![image](https://github.com/user-attachments/assets/f4884a00-5f3f-4391-9b5b-c1d84a070611)


# Notebook Overview

**1. Digital Image Basics**
- **Loading and Displaying an Image:** Demonstrates loading a digital image using skimage and displaying it with matplotlib.
- **Image Features:** Inspecting pixel values and dimensions of the image, converting data types, and performing mathematical operations to modify image brightness.

![image](https://github.com/user-attachments/assets/4d680442-cb69-4337-b675-dc3db7c35783)


**2. Filter Operations on Images**
- **Simple Filter:** A basic operation to darken a specific region of an image using pixel manipulation.

![image](https://github.com/user-attachments/assets/4958092e-2dd8-48a2-b5c1-5c43be1d9309)


**3. AlexNet CNN for MNIST Classification**
- **Model Architecture:** Builds a simplified AlexNet CNN tailored for classifying MNIST images.
- **Training & Testing:** Includes functions to train the model and evaluate its accuracy on test data.
- **Visualization:** Displays images from the MNIST dataset for inspection before training.

![image](https://github.com/user-attachments/assets/6b1eff95-6b8e-491d-b173-f1891ea4b431)


# AlexNet Model Architecture  

The AlexNet architecture in this project has been modified for MNIST classification with the following layers:  
- Convolution Layers: 4 convolutional layers followed by ReLU activations.
- Pooling Layers: Max-pooling layers to reduce the spatial dimensions.
- Fully Connected Layers: 3 fully connected layers to classify MNIST digits into 10 classes.

![image](https://github.com/user-attachments/assets/a2d424c5-f35b-4b17-b425-5a20390de127)


**Model Summary:**

- 93,819,402 trainable parameters.
- Input: Grayscale images (1 channel) of size 28x28.
- Output: Probabilities for 10 digit classes.

# Usage

1. Run the Notebook: Launch the notebook and run each cell sequentially. The training process will output the model's accuracy and loss for each epoch.
2. Train the Model: The model is trained for 2 epochs with a batch size of 16. The optimizer used is SGD with a learning rate of 0.01. The training process outputs loss values and model accuracy.
3. Test the Model: The testing function computes the average loss and accuracy after each epoch.

# Results

```python
epochs = 2

for epoch in range(1, epochs+1):
    train(model,optimizer,train_loader,epoch)
    accuracy = test(model,test_loader)
```

After 2 epochs, the model achieves:

- Epoch 1: 96.76% accuracy on the test set.
- Epoch 2: 98.23% accuracy on the test set.

The results demonstrate how well the adapted AlexNet model performs on the MNIST dataset.


