### README

# Deep Learning Based Melon Ticket CAPTCHA Recognition

This project uses TensorFlow and Keras to build and train neural network models for recognizing 26 letters in CAPTCHA images. The project is improved based on the [Keras Official Example](https://keras.io/examples/vision/captcha_ocr/) and optimized for specific CAPTCHA scenarios.

![captcha_130](https://github.com/lstrhsu/MyHost/blob/main/pics/captcha_130.png)

## Quick Start (Beginner Friendly)

#### 1. Environment Setup

Create and activate conda environment using the following commands:
```bash
conda create -n captcha_ocr python=3.10.13
```
```bash
conda activate captcha_ocr
```
Install required Python packages:
```bash
pip install tensorflow==2.9.0
```
```bash
pip install numpy==1.26.4
```
```bash
pip install matplotlib==3.9.2
```

#### 2. Dataset Collection

You can quickly prepare the dataset using [CaptchaToolkit](https://github.com/lstrhsu/CaptchaToolkit).  
The project provides 78 images for learning and experimentation.  
As a reference, using 860 images as a dataset can achieve the following training results:

```
Training Results Example (Epoch 124):
- Training Loss: 0.0981
- Validation Loss: 0.02192 ↓ (improved from 0.02669)
- Model Status: Saved to model.h5
```

This result indicates:
- Validation loss is significantly lower than training loss, showing no overfitting
- Continuous decrease in validation loss indicates effective learning
- Final validation loss of 0.02192 demonstrates excellent recognition accuracy

![captcha_predict](https://github.com/lstrhsu/MyHost/blob/main/pics/captcha_predict.png)

#### 3. Model Training

Follow the steps in `turtorial.ipynb`, which contains detailed annotations. After manually selecting the kernel, Jupyter Notebook installation is required.

## Principles

#### 1. CNN (Convolutional Neural Network)
- CNN is specialized for processing grid-structured data (like images) and automatically learns image features through convolution operations.
- In this project, CNN extracts visual features from CAPTCHA images, such as edges, textures, and shapes.
- Through multiple layers of convolution and pooling operations, CNN builds feature representations from low to high level.

#### 2. RNN (Recurrent Neural Network)
- RNN excels at processing sequential data and can make more accurate predictions using contextual information.
- In CAPTCHA recognition, RNN improves recognition accuracy by considering character context relationships.
- For example, when recognizing the letter "m", RNN uses contextual information to avoid misidentification as "i" or other similar characters.

#### 3. CTC (Connectionist Temporal Classification)
- CTC is an algorithm for solving sequence labeling problems, particularly suitable for OCR scenarios.
- It automatically aligns input sequences and label sequences without explicit segmentation position information.
- CTC calculates conditional probability P(Y|X) to predict target sequence Y given input sequence X.
- The algorithm introduces blank labels (ε) to handle spacing between characters and merges repeated predictions to output complete text sequences.

## Model Architecture

#### 1. Input Processing
```
Input(shape=(width, height, 1)) → BatchNormalization()
```
- Receives grayscale image input
- Implements data normalization through BatchNormalization

#### 2. Feature Extraction Network
```
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.2)
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.2)
```
- Two convolution blocks extract image features
- Each block includes normalization, activation, and pooling operations
- Dropout layers prevent overfitting

#### 3. Feature Transformation
```
Reshape((width/4, height/4 * 64)) → Dense(128) → BatchNorm → ReLU → Dropout(0.4)
```
- Reshapes feature dimensions for sequence processing
- Feature transformation through fully connected layer
- Higher Dropout rate (0.4) enhances generalization

#### 4. Sequence Recognition
```
Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))
```
- Bidirectional LSTM processes sequence features
- Sequence return mode captures contextual information
- Built-in Dropout mechanism prevents overfitting

#### 5. Output Layer
```
Dense(vocab_size + 1, activation='softmax') → CTCLayer
```
- Softmax layer outputs character probability distribution
- CTC layer handles sequence alignment and loss calculation

#### 6. Optimization Strategy

- **Optimizer**: Adam (learning_rate=0.0005)
- **Loss Function**: CTC Loss
- **Regularization**: L2 regularization

## Detailed Model Structure

| Layer (Type)           | Output Shape         | Params  | Description            |
|-----------------------|---------------------|----------|------------------------|
| image (InputLayer)     | (None, 280, 80, 1)  | 0       | Grayscale image input  |
| BatchNormalization     | (None, 280, 80, 1)  | 4       | Input normalization    |
| Conv1 (Conv2D)         | (None, 280, 80, 32) | 320     | First convolution     |
| BatchNormalization     | (None, 280, 80, 32) | 128     | Feature normalization  |
| Activation (ReLU)      | (None, 280, 80, 32) | 0       | ReLU activation       |
| pool1 (MaxPooling2D)   | (None, 140, 40, 32) | 0       | Feature pooling       |
| Dropout               | (None, 140, 40, 32) | 0       | Prevent overfitting   |
| Conv2 (Conv2D)         | (None, 140, 40, 64) | 18,496  | Second convolution    |
| BatchNormalization     | (None, 140, 40, 64) | 256     | Feature normalization  |
| Activation (ReLU)      | (None, 140, 40, 64) | 0       | ReLU activation       |
| pool2 (MaxPooling2D)   | (None, 70, 20, 64)  | 0       | Feature pooling       |
| Dropout               | (None, 70, 20, 64)  | 0       | Prevent overfitting   |
| reshape (Reshape)      | (None, 70, 1280)    | 0       | Reshape features      |
| dense1 (Dense)         | (None, 70, 128)     | 163,968 | Fully connected layer |
| BatchNormalization     | (None, 70, 128)     | 512     | Feature normalization  |
| Activation (ReLU)      | (None, 70, 128)     | 0       | ReLU activation       |
| Dropout               | (None, 70, 128)     | 0       | Prevent overfitting   |
| Bidirectional (LSTM)   | (None, 70, 128)     | 98,816  | Bidirectional LSTM    |
| label (InputLayer)     | (None, None)        | 0       | Label input           |
| dense2 (Dense)         | (None, 70, 27)      | 3,483   | Output layer          |
| ctc_loss (CTCLayer)    | (None, 70, 27)      | 0       | CTC loss calculation  |

**Total Parameters**: 285,983  
**Trainable Parameters**: 285,533  
**Non-trainable Parameters**: 450

## License
This project is licensed under the MIT License.
