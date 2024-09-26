# ResNet Fine-Tuning & Training on CIFAR100

This repository provides a comprehensive guide on fine-tuning and training the ResNet18 architecture on the CIFAR100 dataset using PyTorch. The project explores three different approaches:
1. **Shallow Fine-Tuning:** Freezing the majority of the network and only training the final fully connected layers.
2. **Deep Fine-Tuning:** Unfreezing the last few layers of the ResNet model and fine-tuning them alongside the final layer.
3. **Training from Scratch:** Training the entire ResNet18 model from scratch without any pre-trained weights.

## Dataset
The model is trained and evaluated on the CIFAR100 dataset, which consists of 60,000 32x32 color images in 100 classes, with 600 images per class. There are 50,000 training images and 10,000 test images.

## Project Structure

- `DataModule`: A data handling class that prepares and loads the CIFAR100 dataset with the necessary transformations (including augmentations for training).
- `DLModel`: A deep learning model class that wraps around the ResNet18 architecture, defining the forward pass, loss calculation, and metrics like accuracy.
- Training Loops: The project contains training and validation loops for all three training approaches:
  - **Shallow Fine-Tuning**
  - **Deep Fine-Tuning**
  - **Training from Scratch**

## Key Features

- **Pre-trained ResNet18 Model**: Fine-tuning is performed on the pre-trained ImageNet model.
- **Data Augmentation**: Various data augmentations are applied to enhance training, including resizing, cropping, and random horizontal flips.
- **Metrics**: Top-1 and Top-5 accuracy metrics are tracked during training and validation.
- **Training and Validation Loss Comparison**: Performance of each approach is compared through loss and accuracy visualizations.

## Results

The notebook compares the training loss and validation accuracy across the three approaches. It uses matplotlib and seaborn to visualize the results.
- **Comparing the Training loss per epoch:** The plot shows the loss reduction per epoch across the three methods.

![image](https://github.com/user-attachments/assets/b7d22e16-26e4-412f-b1d9-db076ed4c98c)

- **Comparing the validation Accuracies per epoch:** This graph compares how well the models generalize to unseen data, tracked across the epochs.

![image](https://github.com/user-attachments/assets/370005da-8262-411f-b32b-ed72e1d639df)




