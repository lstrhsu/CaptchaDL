[English](./README.md) | [中文](./README_zh-CN.md)

# Deep Learning Based Melon Ticket CAPTCHA Recognition

This project uses TensorFlow and Keras to build and train neural network models for recognizing 26 letters in CAPTCHA images. The project is improved based on the [Keras Official Example](https://keras.io/examples/vision/captcha_ocr/) and optimized for specific CAPTCHA scenarios.

![captcha_130](https://github.com/lstrhsu/MyHost/blob/main/pics/captcha_130.png)

## Files
The repository contains the following files:
- `tutorial.ipynb`: Model training tutorial with CTC
- `noCTC_tutorial.ipynb`: Alternative model training tutorial without CTC
- `image.ipynb`: Image processing experiments (for adapting code to other image scenarios)
- `captcha_images`: CAPTCHA image folder
- `userscript`: Folder containing userscript and models (work in progress)

The ultimate goal is to develop a userscript capable of automatic recognition and submission. To enable browser-side prediction, we attempted to convert the Keras model using TFJS, ONNX, and other conversion methods for deployment. However, we encountered some issues ([Known Issues](#known-issues)).  
Currently, the project provides easy-to-reproduce model training for corresponding CAPTCHAs with simple tutorials.

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
Epoch 58/150
48/49 [============================>.] - ETA: 0s - loss: 0.9573
Epoch 58: val_loss improved from 0.57060 to 0.48835, saving model to model.h5
```
```
Epoch 124/150
48/49 [============================>.] - ETA: 0s - loss: 0.0981 
Epoch 124: val_loss improved from 0.02669 to 0.02192, saving model to model.h5
```
In actual prediction, the accuracy remains as high as 99.99%.

![captcha_predict](https://github.com/lstrhsu/MyHost/blob/main/pics/captcha_predict.png)

#### 3. Model Training

Follow the steps in `tutorial.ipynb`, which contains detailed annotations. After manually selecting the kernel, Jupyter Notebook installation is required.

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
- CTC automatically aligns input sequences and label sequences without explicit segmentation position information.
- CTC calculates conditional probability P(Y|X) to predict target sequence Y given input sequence X.
- CTC introduces blank labels (ε) to handle spacing between characters and merges repeated predictions to output complete text sequences.

## Model Architecture
**The following introduction is based on `tutorial.ipynb`.**

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

#### 7. Detailed Model Structure

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

## Known Issues
1. CTC Loss
Unfortunately, when using the model in JavaScript, neither TensorFlow.js nor ONNX.js has native implementation of CTC loss. So if you want to make predictions in the browser backend, you may need to reconsider the model architecture.  
This is also why I created a second model without the CTC layer, which you can find in `noCTC_tutorial.ipynb`.

2. `noCTC_tutorial.ipynb`
This is an alternative solution without using CTC layer. This approach requires a larger dataset. As a reference, 13,000 images can achieve a val_loss less than 2 after 200 epochs of training.  
Since I haven't done too detailed research, it might not be the best model structure. So here's just a brief introduction to its implementation:
- Split the 6-digit CAPTCHA into 6 independent character recognition tasks
- Process images uniformly to 80×280 grayscale images
- Use 3 CNN convolution blocks to extract image features
- Use categorical_crossentropy as loss function

3. TensorFlow.js
Following `noCTC_tutorial.ipynb`, I created two `HDF5` model files and trained them in the same way, with the only difference being the dataset size, resulting in different val_loss values.  
For the test sample "TARJOT", the results are as follows:
- TensorFlow 2.15.0 (Keras 3): val_loss ≈ 2.0 Prediction: RBRQMQ
- TensorFlow 2.9.0 (Keras 2): val_loss ≈ 6.0 Prediction: TMRBQZ

## Others
This project is licensed under the MIT License.  

For issues or suggestions, please submit an Issue. If you have the intention to help complete the remaining parts of the project, please contact me at any time.  
This tool is for educational and research purposes only. Users should comply with the website's terms of service and relevant laws and regulations. The authors are not responsible for any misuse or potential consequences of using this tool.
