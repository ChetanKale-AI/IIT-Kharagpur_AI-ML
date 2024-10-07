# Mistral 7B Text Generation and Function Calling Demonstration

This repository contains a Jupyter notebook showcasing the use of the Mistral 7B model for text generation, zero-shot and few-shot prompting, and function calling with external APIs or user-defined functions. The notebook demonstrates how to create a pipeline for loading a pre-trained model from Hugging Face and using it to perform tasks such as text generation, mathematical calculations, weather information retrieval, and more.

# Introduction

This project uses the Mistral 7B model from Hugging Face for various natural language processing tasks. It also implements function calling, allowing the model to execute Python functions during the conversation. This capability can be extended to external APIs, making the model interactive and versatile.

# Requirements
- Python 3.8 or above
- Jupyter Notebook
- Hugging Face transformers library
- accelerate for distributed model loading
- torchvision for vision-related tasks (if necessary)
- sentencepiece for tokenization
- Optional: Any external API libraries if extending function calling (e.g., requests for weather APIs)

# Installation

To run the notebook, you need to install the necessary dependencies. You can install them via pip:  

```python
pip install --upgrade jinja2
pip install 'transformers[sentencepiece]' accelerate torchvision
```
Ensure you have the necessary API tokens and keys (e.g., Hugging Face API token for model access).

# Notebook Overview

The notebook is structured as follows:

- **Model Setup:** Loading the Mistral 7B model and tokenizer from Hugging Face using an API token.
- **Text Generation:** Implementing a text generation pipeline with various configuration options such as temperature, top-k sampling, and top-p filtering.
- **Prompting Techniques:** Examples of zero-shot, few-shot, and chain-of-thought prompting to handle complex tasks like math problem solving.
- **Function Calling:** Demonstrating how the model can generate function arguments, invoke a Python function (e.g., getting current temperature), and handle the result.
- **Tokenization:** An example of how input text is tokenized and converted into token IDs.

# Function Calling

One of the most interesting features demonstrated in this notebook is function calling. This allows the model to generate arguments for Python functions during inference and execute those functions dynamically. This capability makes the model interactive and capable of performing real-world tasks like retrieving weather data or making calculations.

For example, we define two functions:

- get_current_temperature(location, unit) returns a default temperature based on the location and unit provided.
- get_current_wind_speed(location) returns a default wind speed.

The model generates the required arguments for these functions and outputs the results based on the conversation context.

# Tokenization Example

The notebook includes an example of how text is tokenized and converted to token IDs. This is important for understanding how models like Mistral handle input data at a lower level.
```python
input = "This prompt engineering course is offered by AI4ICPS."
model_inputs = tokenizer([input], return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])
```

# Model Prompting Examples

We explore three types of prompting:

- **Zero-shot Prompting:** The model answers a math problem without prior examples.
- **Few-shot Prompting:** The model answers based on a couple of provided examples.
- **Chain-of-Thought Prompting:** The model solves a math problem by explaining its reasoning step-by-step.

# Conclusion

This notebook demonstrates how to leverage the Mistral 7B model for various natural language tasks, from simple text generation to dynamic function calling. You can use this as a foundation to extend the model’s capabilities for more complex and domain-specific applications, such as API integration, interactive assistants, and advanced problem-solving.

Feel free to explore the notebook and adapt the code to suit your use cases.
