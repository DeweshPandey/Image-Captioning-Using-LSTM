# Image Captioning with CNN-RNN Architecture

This repository contains an implementation of an image captioning system built using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The model generates captions for images by extracting visual features using a pre-trained Inception V3 model and generating textual descriptions using an LSTM decoder.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Implementation Details](#implementation-details)
5. [Setup and Installation](#setup-and-installation)
6. [Training](#training)
7. [Results](#results)
8. [How to Use](#how-to-use)
9. [References](#references)

---

## Introduction
Image captioning is a task that combines computer vision and natural language processing. This project uses:
- **Inception V3** (pre-trained on ImageNet) for feature extraction.
- **LSTM** for sequence generation of captions.

The model:
- Encodes image features using the CNN.
- Generates captions word by word using the RNN, with teacher forcing during training and greedy decoding during inference.

---

## Dataset
The [Flickr8k dataset](https://www.kaggle.com/adityajn105/flickr8k) is used for training and evaluation. It consists of:
- **8,000 images**, each with 5 corresponding captions.
- Annotations in a `.csv` file.

---

## Model Architecture
1. **Encoder**:
   - Uses Inception V3 with the fully connected layers replaced by a linear layer to produce feature embeddings.
   - Gradients are disabled for all layers except the final linear layer.

2. **Decoder**:
   - Uses an embedding layer to convert words into dense vectors.
   - An LSTM generates captions based on the image features and word embeddings.
   - Outputs are passed through a linear layer to predict the next word.

3. **Vocabulary**:
   - Built using SpaCy for tokenization.
   - Contains special tokens: `<PAD>`, `<SOS>`, `<EOS>`, and `<UNK>`.

---

## Implementation Details
- **Framework**: PyTorch
- **Preprocessing**:
  - Images are resized to 299x299 and normalized.
  - Captions are tokenized and converted to indices.
  - Batches are padded for consistent input lengths.
- **Loss Function**:
  - CrossEntropyLoss with padding ignored.
- **Optimizer**:
  - Adam optimizer with a learning rate of 3e-2.

---

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-captioning
   cd image-captioning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset using Kaggle API:
   ```bash
   kaggle datasets download -d adityajn105/flickr8k
   ```
4. Extract the dataset:
   ```bash
   unzip flickr8k.zip -d data/
   ```

---

## Training
Run the training script:
```bash
python train.py
```
Hyperparameters:
- Embedding size: 256
- Hidden size: 256
- Batch size: 32
- Number of epochs: 10

During training:
- Training loss and accuracy are logged to TensorBoard.

---

## Results
- Model learns to generate meaningful captions for images.
- Example:
  - **Input Image**:
    ![Example Image](path/to/image.jpg)
  - **Generated Caption**: "A dog running in a grassy field."

---

## How to Use
1. Load a trained model checkpoint:
   ```python
   from model import CNNtoRNN
   model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
   model.load_state_dict(torch.load("my_checkpoint.pth.tar"))
   ```
2. Generate captions:
   ```python
   caption = model.caption_image(image, vocabulary)
   print("Generated Caption:", " ".join(caption))
   ```

---

## References
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Inception V3 Paper](https://arxiv.org/abs/1512.00567)
- [Flickr8k Dataset](https://www.kaggle.com/adityajn105/flickr8k)

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

