# Introduction to Deep Learning – Assignments

## Overview

This repository contains assignments completed as part of the **Introduction to Deep Learning** course. The assignments explore the implementation, training, and evaluation of different deep learning architectures using **Python and PyTorch**.

The work covers fundamental neural networks, image classification, regression, convolutional neural networks (CNNs), recurrent neural networks (RNNs), LSTMs, GRUs, sequence-to-sequence models, attention mechanisms, and Transformer architectures.

The assignments focus on understanding how different neural network architectures process data, learn patterns, and generate predictions across computer vision and natural language processing (NLP) tasks.

## Technologies and Libraries

- **Python** – Programming language.
- **PyTorch** – Neural network implementation, training, and evaluation.
- **Torchvision** – Image datasets and transformations.
- **NumPy** – Numerical computations and array manipulation.
- **Pandas** – Dataset loading and preprocessing.
- **Scikit-learn** – Data splitting, preprocessing, and evaluation metrics.
- **Matplotlib** – Training and validation visualization.
- **Seaborn** – Confusion matrix visualization.

---

## Assignment 1 – Neural Networks and Regression

**Notebook:** `Assignment1.ipynb`

This assignment explores the implementation of Multi-Layer Perceptrons (MLPs) for image classification and regression tasks using PyTorch.

### Problem 1: Image Classification using MLPs

**Dataset:** CIFAR-10

Implemented fully connected neural networks to classify images into the ten CIFAR-10 categories.

The assignment includes:

- Loading and exploring the CIFAR-10 dataset using Torchvision.
- Calculating image channel means and standard deviations.
- Normalizing image data.
- Preparing training and testing DataLoaders.
- Implementing an MLP architecture with multiple fully connected layers and ReLU activation functions.
- Implementing a second, deeper MLP architecture to experiment with network depth and hidden layer dimensions.
- Training models using Cross-Entropy Loss and Stochastic Gradient Descent (SGD).
- Evaluating classification performance.

**Evaluation and Analysis:**
- Training and validation loss.
- Training and validation accuracy.
- Precision, recall, and F1-score.
- Confusion matrix visualization.
- Trainable parameter count.
- Comparison of different MLP architectures.

### Problem 2: House Price Prediction using MLPs

**Dataset:** House Prices Training Dataset (`house-train.csv`)

Implemented neural network regression models to predict house sale prices using property-related features.

**Problem 2a – Numerical Feature Processing**

- Loading the housing dataset.
- Selecting relevant numerical features.
- Handling missing values using mean imputation.
- Splitting data into training and validation sets.
- Scaling numerical features.
- Converting processed data into PyTorch tensors.
- Building an MLP regression model.
- Training using Mean Squared Error (MSE) loss and the Adam optimizer.

**Problem 2b – Numerical and Categorical Features**

Extended the regression implementation by incorporating categorical features.

- Identifying numerical and categorical columns.
- Handling missing numerical values using mean imputation.
- Handling missing categorical values using mode imputation.
- Applying one-hot encoding to categorical features.
- Splitting and scaling the processed dataset.
- Implementing an MLP regression architecture.
- Training and evaluating the model.

**Problem 2c – Additional MLP Regression Experiment**

Implemented another MLP architecture to further explore neural network-based house price prediction using the processed dataset.

**Evaluation and Analysis:**
- Training and validation loss.
- Regression performance analysis.
- Loss curve visualization.
- Model parameter count.
- Comparison of feature processing and model configurations.

---

## Homework 2 – Convolutional Neural Networks

**Notebook:** `Homework2.ipynb`

This assignment focuses on convolutional neural networks for image classification using CIFAR-10. It explores CNN architectures, regularization, and the effect of dropout on model performance.

### Part 1: AlexNet Implementation

Implemented an AlexNet-inspired CNN architecture for CIFAR-10 image classification.

**Implementation:**
- Loading and preprocessing CIFAR-10 images.
- Calculating channel-wise mean and standard deviation.
- Applying image normalization.
- Preparing training and testing DataLoaders.
- Implementing convolutional layers with ReLU activation.
- Applying max-pooling operations.
- Implementing fully connected classification layers.
- Training the CNN using Cross-Entropy Loss and SGD.

**Dropout Experiment:**

Two AlexNet configurations were implemented:

1. AlexNet without dropout.
2. AlexNet with dropout (dropout probability = 0.5).

The experiment examines how dropout affects model generalization and the difference between training and validation loss.

### Part 2: ResNet-18 Implementation

Implemented a ResNet-18 architecture using residual learning.

**Implementation:**
- Developing residual Basic Blocks.
- Implementing convolutional layers and batch normalization.
- Applying ReLU activation.
- Implementing shortcut connections to support residual learning.
- Constructing ResNet-18 using four residual stages.
- Training the network for CIFAR-10 classification.

**Dropout Experiment:**

Compared ResNet-18 configurations with and without dropout to examine the impact of regularization.

**Evaluation and Analysis:**
- Training and validation loss.
- Classification performance.
- Model parameter count.
- Comparison of AlexNet and ResNet-18.
- Analysis of dropout and its influence on generalization.

---

## Homework 3 – Recurrent Neural Networks and Sequence Modeling

**Notebook:** `Homework3.ipynb`

This assignment explores recurrent neural network architectures for sequential data processing and next-character prediction.

### Problem 1: Next Character Prediction

Implemented three recurrent neural network architectures:

- Vanilla RNN-based model.
- Long Short-Term Memory (LSTM).
- Gated Recurrent Unit (GRU).

**Implementation:**
- Preparing a text corpus for character-level prediction.
- Creating character-to-index and index-to-character mappings.
- Converting text into numerical sequences.
- Preparing input sequences and their corresponding next-character targets.
- Implementing embedding layers.
- Building RNN, LSTM, and GRU architectures.
- Training models using Cross-Entropy Loss.
- Evaluating predictions on validation data.
- Generating next-character predictions from input text.

**Sequence Length Experiments:**

Experiments were performed using different input sequence lengths:

- 10 characters.
- 20 characters.
- 30 characters.

The objective is to understand how the length of input sequences influences the ability of recurrent networks to learn contextual dependencies.

### Problem 2: Character-Level Language Modeling

Extended sequence modeling experiments using the Tiny Shakespeare text dataset.

**Implementation:**
- Downloading the Tiny Shakespeare dataset.
- Building character-level vocabulary mappings.
- Encoding text into integer sequences.
- Creating a custom PyTorch Dataset and DataLoader.
- Implementing LSTM and GRU language models.
- Training and validating the models.
- Generating text predictions.

**Additional Experiments:**
- Sequence lengths of 20, 30, and 50.
- Increasing hidden dimensions and the number of recurrent layers.
- Comparing LSTM and GRU performance.
- Measuring validation loss and prediction accuracy.
- Calculating model size and trainable parameter count.
- Measuring inference time.
- Comparing model behavior across different sequence lengths.

---

## Homework 4 – Sequence-to-Sequence Learning and Attention Mechanisms

**Notebook:** `Intro_to_DL_HW4.ipynb`

This assignment focuses on neural machine translation using encoder-decoder architectures and attention mechanisms.

The primary task is English-to-French translation, followed by French-to-English translation.

### Problem 1: English-to-French Translation Without Attention

Implemented a sequence-to-sequence (Seq2Seq) model using an encoder-decoder architecture.

**Implementation:**
- Preparing English-French sentence pairs.
- Creating word-to-index and index-to-word mappings.
- Introducing Start of Sequence (SOS) and End of Sequence (EOS) tokens.
- Building a custom translation dataset.
- Implementing the encoder network.
- Implementing the decoder without an attention mechanism.
- Training the encoder-decoder architecture.
- Generating French translations from English input.

### Problem 2: English-to-French Translation With Attention

Extended the Seq2Seq implementation by incorporating an attention mechanism.

**Implementation:**
- Reusing the English-French translation dataset.
- Implementing the encoder.
- Developing an attention-based decoder.
- Calculating attention weights over encoder outputs.
- Incorporating relevant encoder information into decoder predictions.
- Training and evaluating the attention-based translation model.

The experiment explores how attention helps the decoder access relevant information from different input positions instead of relying only on the final encoder representation.

### Problem 3: French-to-English Translation

Extended the translation task by reversing the source and target languages.

Two approaches were implemented:

1. French-to-English translation without attention.
2. French-to-English translation with attention.

**Evaluation and Analysis:**
- Training and validation behavior.
- Translation output generation.
- Comparison of encoder-decoder models with and without attention.
- Analysis of translation performance in both language directions.

---

## Homework 5 – Transformer Architectures

The Homework 5 assignments explore Transformer-based architectures for character-level language modeling and neural machine translation.

### Problem 1: Character-Level Next-Character Prediction Using Transformers

**Notebooks:** `DL_HW5_P1.ipynb` and `HW5/DL_HW5_P1.ipynb`

Implemented a Transformer-based model for next-character prediction.

**Implementation:**
- Preparing a text corpus for character-level prediction.
- Creating character vocabulary mappings.
- Converting text into numerical sequences.
- Preparing input sequences and next-character targets.
- Implementing a Transformer-based neural network using PyTorch.
- Training and validating the model.
- Generating next-character predictions.
- Calculating trainable parameter counts.

**Sequence Length Experiments:**

Compared different input sequence lengths:

- 10 characters.
- 20 characters.
- 30 characters.

The experiments investigate how input context length influences Transformer-based character prediction.

### Problem 2: Character-Level Language Modeling Using Transformers

**Notebook:** `HW5/DL_HW5_P2 (1).ipynb`

Implemented a Transformer-based language model using the Tiny Shakespeare dataset.

**Implementation:**
- Downloading the Tiny Shakespeare dataset.
- Preparing character-level vocabulary mappings.
- Encoding the complete text corpus into integer sequences.
- Creating a custom PyTorch Dataset and DataLoader.
- Implementing a Transformer-based character prediction model.
- Training the model to predict the next character.
- Generating text predictions.
- Measuring model parameters and computational performance.

**Experiments:**

1. Sequence length of 20.
2. Sequence length of 30.
3. Sequence length of 50.
4. Increasing hidden dimensions and the number of Transformer layers.

The experiments examine how sequence length and model capacity influence language modeling performance.

### Problem 3: English-to-French Translation Using Transformers

**Notebook:** `HW5/DL_HW5_P3.ipynb`

Implemented a Transformer-based neural machine translation model for English-to-French translation.

**Implementation:**
- Preparing English-French sentence pairs.
- Creating word-level vocabulary mappings.
- Introducing SOS and EOS tokens.
- Implementing a custom translation Dataset.
- Preparing batched training data using DataLoader.
- Implementing the Transformer architecture.
- Training the model for sequence-to-sequence translation.
- Evaluating generated translations.

**Evaluation:**
- Translation loss.
- Prediction accuracy.
- Generated translation examples.
- Comparison between target and predicted sequences.

### Problem 4: French-to-English Translation Using Transformers

**Notebook:** `HW5/DL_HW5_P4.ipynb`

Extended the Transformer-based translation implementation to perform French-to-English translation.

**Implementation:**
- Reversing the English-French sentence pairs.
- Preparing French-to-English vocabulary mappings.
- Implementing the translation dataset.
- Preparing input and target sequences.
- Training the Transformer architecture.
- Generating English translations from French input.
- Evaluating model predictions.

**Evaluation:**
- Translation loss.
- Prediction accuracy.
- Generated translation examples.
- Analysis of translation performance in the reverse language direction.

---

## Key Learning Outcomes

Through these assignments, the following concepts and practical skills were explored:

- Designing and implementing neural networks using PyTorch.
- Understanding forward propagation and backpropagation.
- Training models using different loss functions and optimizers.
- Understanding image normalization and dataset preprocessing.
- Implementing MLPs for classification and regression.
- Understanding convolutional operations and pooling.
- Implementing residual connections and ResNet architectures.
- Applying dropout and batch normalization.
- Understanding sequential data processing.
- Implementing RNN, LSTM, and GRU architectures.
- Performing character-level language modeling.
- Understanding encoder-decoder architectures.
- Implementing attention mechanisms for machine translation.
- Exploring Transformer architectures.
- Comparing different neural network architectures and configurations.
- Evaluating models using loss, accuracy, precision, recall, and F1-score.
- Analyzing training behavior, model complexity, parameter counts, and inference time.

## Conclusion

These assignments provide practical experience with fundamental and advanced deep learning architectures, progressing from fully connected neural networks to convolutional, recurrent, attention-based, and Transformer models.

The implementations demonstrate the application of deep learning techniques to computer vision, regression, character-level language modeling, and neural machine translation using PyTorch.

The repository serves as a collection of hands-on experiments and implementations developed while studying the concepts of Introduction to Deep Learning.
