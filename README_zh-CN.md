### README_zh-CN

# 基于深度学习的Melon Ticket验证码识别

本项目使用TensorFlow和Keras构建和训练神经网络模型，识别验证码图像中的26个字母。本项目基于[Keras官方示例](https://keras.io/examples/vision/captcha_ocr/)进行改进，针对特定验证码场景进行了优化。

![captcha_130](https://github.com/lstrhsu/MyHost/blob/main/pics/captcha_130.png)

## 快速使用（新手友好）

#### 1. 配置环境

使用以下命令创建并激活conda环境：
```bash
conda create -n captcha_ocr python=3.10.13
```
```bash
conda activate captcha_ocr
```
安装所需的Python包：
```bash
pip install tensorflow==2.9.0
```
```bash
pip install numpy==1.26.4
```
```bash
pip install matplotlib==3.9.2
```

#### 2. 收集数据集

可以通过[CaptchaToolkit](https://github.com/lstrhsu/CaptchaToolkit)快速准备数据集。  
项目文件中提供了78张图片以便于学习和尝试。  
作为参考，使用860张图片作为数据集可以达到以下训练效果：
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
实际预测中，准确率依旧高达99.99%。
![captcha_predict](https://github.com/lstrhsu/MyHost/blob/main/pics/captcha_predict.png)

#### 3. 训练模型

根据`turtorial.ipynb`逐步运行，文件中包含详细的注解。手动选择内核后，需要安装Jupyter Notebook。

## 原理

#### 1. CNN（卷积神经网络）
- CNN专门用于处理具有网格结构的数据（如图像），通过卷积操作自动学习图像的特征。
- 在本项目中，CNN负责提取验证码图像的视觉特征，如边缘、纹理和形状等。
- 通过多层卷积和池化操作，CNN能够逐层构建从低级到高级的特征表示。

#### 2. RNN（循环神经网络）
- RNN擅长处理序列数据，能够利用上下文信息做出更准确的预测。
- 在验证码识别中，RNN通过考虑字符的上下文关系来提高识别准确率。
- 例如，当识别字母"m"时，RNN会结合前后文信息，避免将其误识别为"i"或其他相似字符。

#### 3. CTC（连接主义时间分类）
- CTC是一种解决序列标注问题的算法，特别适用于OCR等场景。
- 它能够自动对齐输入序列和标签序列，无需明确的分割位置信息。
- CTC通过计算条件概率P(Y|X)，在给定输入序列X的情况下预测目标序列Y。
- 算法会引入空白标记(ε)来处理字符之间的间隔，并合并重复预测，最终输出完整的文本序列。

## 模型架构

#### 1. 输入处理
```
Input(shape=(width, height, 1)) → BatchNormalization()
```
- 接收灰度图像输入
- 通过BatchNormalization实现数据归一化

#### 2. 特征提取网络
```
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.2)
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.2)
```
- 两个卷积块提取图像特征
- 每个卷积块包含标准化、激活和池化操作
- Dropout层防止过拟合

#### 3. 特征转换
```
Reshape((width/4, height/4 * 64)) → Dense(128) → BatchNorm → ReLU → Dropout(0.4)
```
- 重塑特征维度以适应序列处理
- 通过全连接层进行特征转换
- 较高的Dropout率(0.4)增强泛化能力

#### 4. 序列识别
```
Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))
```
- 双向LSTM处理序列特征
- 序列返回模式捕获上下文信息
- 内置Dropout机制防止过拟合

#### 5. 输出层
```
Dense(vocab_size + 1, activation='softmax') → CTCLayer
```
- Softmax层输出字符概率分布
- CTC层处理序列对齐和损失计算

#### 6. 优化策略

- **优化器**: Adam (learning_rate=0.0005)
- **损失函数**: CTC Loss
- **正则化**: L2 regularization

## 详细模型结构

| 层 (类型)                | 输出形状              | 参数量  | 说明                    |
|------------------------|---------------------|---------|------------------------|
| image (InputLayer)     | (None, 280, 80, 1)  | 0       | 灰度图像输入              |
| BatchNormalization     | (None, 280, 80, 1)  | 4       | 输入数据归一化            |
| Conv1 (Conv2D)         | (None, 280, 80, 32) | 320     | 第一层卷积               |
| BatchNormalization     | (None, 280, 80, 32) | 128     | 特征归一化               |
| Activation (ReLU)      | (None, 280, 80, 32) | 0       | ReLU激活函数            |
| pool1 (MaxPooling2D)   | (None, 140, 40, 32) | 0       | 特征池化                 |
| Dropout               | (None, 140, 40, 32) | 0       | 防止过拟合               |
| Conv2 (Conv2D)         | (None, 140, 40, 64) | 18,496  | 第二层卷积               |
| BatchNormalization     | (None, 140, 40, 64) | 256     | 特征归一化               |
| Activation (ReLU)      | (None, 140, 40, 64) | 0       | ReLU激活函数            |
| pool2 (MaxPooling2D)   | (None, 70, 20, 64)  | 0       | 特征池化                 |
| Dropout               | (None, 70, 20, 64)  | 0       | 防止过拟合               |
| reshape (Reshape)      | (None, 70, 1280)    | 0       | 重塑特征维度              |
| dense1 (Dense)         | (None, 70, 128)     | 163,968 | 全连接层                |
| BatchNormalization     | (None, 70, 128)     | 512     | 特征归一化               |
| Activation (ReLU)      | (None, 70, 128)     | 0       | ReLU激活函数            |
| Dropout               | (None, 70, 128)     | 0       | 防止过拟合               |
| Bidirectional (LSTM)   | (None, 70, 128)     | 98,816  | 双向LSTM处理            |
| label (InputLayer)     | (None, None)        | 0       | 标签输入                |
| dense2 (Dense)         | (None, 70, 27)      | 3,483   | 输出层                 |
| ctc_loss (CTCLayer)    | (None, 70, 27)      | 0       | CTC损失计算             |

**总参数量**: 285,983  
**可训练参数**：285,533  
**不可训练参数**：450

## 许可证
本项目采用MIT许可证。