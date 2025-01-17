# Image Captioning with InceptionV3 and LSTM

This repository contains a deep learning model for generating captions for images using the **Flickr8k dataset**. The model combines a **pre-trained Inception V3 model** for feature extraction and an **LSTM-based decoder** for generating textual captions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Usage](#usage)
7. [Results](#results)
8. [License](#license)

## Project Overview

This project uses **Inception V3**, a popular CNN architecture, to extract image features and feed them into an **LSTM-based sequence generation model** to generate captions. The approach utilizes:

- **Feature extraction** via a custom Encoder class based on Inception V3 (with pre-trained weights).
- **Sequence generation** using an LSTM network, which generates captions word by word.
- **Teacher forcing** during training to improve convergence.

## Dependencies

To run this project, you need the following Python libraries:

- `torch` (PyTorch)
- `torchvision`
- `torchsummary`
- `spacy`
- `Pandas`
- `PIL`
- `kagglehub`

# Install dependencies using `pip`:

bash
pip install torch torchvision torchsummary spacy pandas pillow kagglehub
Also, download the required SpaCy language model:

bash
Copy
Edit
python -m spacy download en_core_web_sm
Dataset
The model is trained on the Flickr8k dataset, which contains 8,000 images and corresponding captions. You can download the dataset via KaggleHub:

python
Copy
Edit
import kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")
The dataset contains two main files:

Images folder containing the images.
Captions file containing image IDs and associated captions.
Model Architecture
EncoderCNN (Inception V3): Uses the Inception V3 model (without the fully connected layers) for feature extraction, outputting an embedding vector for each image.
DecoderRNN (LSTM): The features extracted by the encoder are passed to an LSTM network. The LSTM generates word sequences for captions.
Vocabulary: Built using SpaCy to map words to indices and prepare data for the model.
CNNtoRNN: Combines the Encoder and Decoder into a single architecture.
Training Process
Data Preprocessing: The images are resized, cropped, and normalized using torchvision.transforms. Captions are tokenized and numericalized using SpaCy.
Model Training: The model is trained for 10 epochs using CrossEntropyLoss. During training, the actual caption is fed to the LSTM (teacher forcing). During testing, the predicted word is fed back as input for the next word prediction.
Hyperparameters:
embed_size = 256
hidden_size = 256
num_layers = 1
learning_rate = 3e-2
batch_size = 32
num_epochs = 10
Usage
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning
Run the training script:

bash
Copy
Edit
python train.py
Use the trained model for inference (captions generation):

python
Copy
Edit
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
caption = model.caption_image(image, dataset.vocab)
print("Generated Caption: ", caption)
Results
The model’s accuracy and loss are logged via TensorBoard during training. You can visualize the metrics by running:

bash
Copy
Edit
tensorboard --logdir=runs
License
This project is licensed under the MIT License.

csharp
Copy
Edit

You can copy-paste this into a `README.md` file in your GitHub repository.
