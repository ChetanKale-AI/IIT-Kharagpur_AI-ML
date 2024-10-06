# Sentiment Analysis using BERT on IMDB Dataset

This notebook demonstrates a Sentiment Analysis task using the BERT (Bidirectional Encoder Representations from Transformers) model, applied to the IMDB movie reviews dataset. The notebook uses the Hugging Face transformers library to load a pre-trained BERT model for fine-tuning on the sentiment classification task.

**Key Features**

- **Data Preprocessing:** The raw text data is cleaned using a pre-processing function, which removes HTML tags, URLs, and extra spaces. The text is also tokenized using BERT's tokenizer.
- **BERT Fine-Tuning:** The BERT model is fine-tuned for a binary classification task, where it learns to predict whether a movie review is positive or negative.
- **Freezing Layers for Transfer Learning:** Layers in BERT are selectively frozen, allowing us to fine-tune only specific layers, thus speeding up the training process and preventing overfitting.
- **Accuracy and Loss Evaluation:** The model is evaluated on the validation set after every epoch, and both accuracy and loss metrics are reported for the training and validation phases.
- **GPU Support:** The notebook detects and utilizes the GPU (if available) for faster training.

# Requirements
The following libraries are required to run the notebook:

- transformers
- torch
- torchmetrics
- datasets
- pandas
- numpy
- tqdm

```python
!pip install transformers
!pip install torchmetrics
!pip install datasets
```
# Dataset

The dataset used in this notebook is the IMDB movie reviews dataset, which can be loaded directly from the Hugging Face datasets library. It consists of 50,000 movie reviews labeled as positive or negative.

# Preprocessing Steps

1. Remove HTML tags, URLs, and extra spaces from the text.
2. Tokenize the text using BERT's tokenizer.
3. Pad and truncate the sequences to a maximum length of 140 tokens.
4. Create attention masks to distinguish between actual tokens and padding tokens.

# Model Architecture

- **Pretrained BERT:** A pre-trained BERT model (bert-base-uncased) is used for fine-tuning.
- **Sequence Classification:** The last layer of the BERT model is replaced with a binary classification head for the sentiment analysis task.
- **Frozen Layers:** All layers except for the classifier and the last two encoder layers are frozen during training to perform transfer learning.

# Training and Evaluation
- **Optimizer:** The model is optimized using the AdamW optimizer with a learning rate of 3e-4.
- **Loss Function:** Cross-entropy loss is used to train the model.
- **Accuracy:** Training and validation accuracy are measured using torchmetrics.Accuracy.

# Usage

**1. Data Preprocessing and Tokenization**

The notebook first preprocesses the raw text data and tokenizes it using BERT's tokenizer. The tokenized data is converted into input IDs and attention masks, which are necessary for the BERT model.

**2. Training**

The model is trained using the IMDB training set, and accuracy and loss metrics are tracked for each epoch.

**3. Evaluation**

After each epoch, the model is evaluated on the validation set. Validation loss and accuracy are printed to monitor the model's performance.

# Results
The notebook achieves a training accuracy of 88.06% and a validation accuracy of 86.38% after 10 epochs of fine-tuning.

