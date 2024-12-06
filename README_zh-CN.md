[English](./README.md) | [中文](./README_zh-CN.md)

### README_zh-CN

# 基于深度学习的Melon Ticket验证码识别

本项目使用TensorFlow和Keras构建和训练神经网络模型，识别验证码图像中的26个字母。本项目基于[Keras官方示例](https://keras.io/examples/vision/captcha_ocr/)进行改进，针对特定验证码场景进行了优化。

![captcha_130](https://github.com/lstrhsu/MyHost/blob/main/pics/captcha_130.png)

## 文件
整个存储库包含以下文件：
- `tutorial.ipynb`：含CTC的训练模型教程
- `noCTC_tutorial.ipynb`：不含CTC的训练模型教程
- `image.ipynb`：对于图像处理方法的实验（便于将代码应用于其他图像场景）
- `captcha_images`：验证码图片文件夹
- `userscript`：包含用户脚本和相应模型的文件夹（未完成）

该项目的最终目的是为了完成一个能够自动识别并提交的用户脚本。为了能够在浏览器后端进行预测，需要对Keras的模型进行转换，因此采用了TFJS、ONNX等转换方式并部署。但遇到了一些问题（详见[已知问题](#已知问题)）。  
所以目前这个项目只有能够轻松复现相对应验证码的模型训练，并制作了简易的教程。

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

根据`tutorial.ipynb`逐步运行，文件中包含详细的注解。手动选择内核后，需要安装Jupyter Notebook。


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
- CTC能够自动对齐输入序列和标签序列，无需明确的分割位置信息。
- CTC通过计算条件概率P(Y|X)，在给定输入序列X的情况下预测目标序列Y。
- CTC会引入空白标记(ε)来处理字符之间的间隔，并合并重复预测，最终输出完整的文本序列。

## 模型架构
**以下介绍是基于`tutorial.ipynb`的。**

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

#### 7. 详细模型结构

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

## 已知问题
#### 1. CTC loss
不幸的是，在JavaScript使用模型的情况下，无论是TensorFlow.js还是ONNX.js，都缺乏CTC损失的原生实现。所以如果你想要在浏览器后端进行预测，可能需要重新考虑模型架构。  
这也就是为什么我制作了第二个没有CTC层的模型的原因，具体内容可以在`noCTC_tutorial.ipynb`中查看。

#### 2. `noCTC_tutorial.ipynb`

这是一个不使用CTC层的替代方案。该方案需要较大的数据集，作为参考，13000张图片在训练200轮后可以达到val_loss小于2的效果。  
由于我并没有做太详细的研究，它可能并不是那么好的模型结构。所以这里只是简要介绍一下它的实现： 
- 将6位验证码拆分为6个独立的字符识别任务，每个字符都有自己的预测分支，每个分支负责预测26个英文字母中的一个。
- 图片统一处理为80×280的灰度图像，每个字符用26维向量表示（对应A-Z）。
- 使用3个CNN卷积块提取图像特征，每个卷积块包含：批归一化、ReLU激活、最大池化和Dropout，最后通过6个独立分支预测每个位置的字符。
- 使用categorical_crossentropy作为损失函数，采用BatchNormalization和Dropout防止过拟合，学习率会根据验证集表现自动调整。

#### 3. TensorFlow.js
按照`noCTC_tutorial.ipynb`，我制作了两个`HDF5`模型文件，并以相同的方式进行训练，唯一的区别就是数据集大小，导致val_loss值不同。  
针对于测试样本"TARJOT"的结果如下：  
- TensorFlow 2.15.0 (Keras 3): val_loss ≈ 2.0 预测结果：RBRQMQ  
- TensorFlow 2.9.0 (Keras 2): val_loss ≈ 6.0 预测结果：TMRBQZ

有趣的是，尽管较新版本的模型具有更低的验证损失（理论上应该表现更好），但在JavaScript环境中的预测效果反而不如验证损失更高的旧版本模型。这两个模型在Python环境中都能正确识别测试样本，但转换到JavaScript后表现却大幅下降。  
这种异常现象可能源于模型转换过程中的问题，或是TensorFlow.js转换器本身的局限性。我对这个方向了解甚少，所以没有再进行尝试。

## 其他
本项目采用MIT许可证。

如有问题或建议，请提交 Issue。如果你有意愿协助完成该项目的剩余部分，请随时联系我。  
本工具仅供教育和研究目的使用。用户应遵守网站的服务条款以及相关法律法规。作者不对工具的滥用或使用后可能产生的任何后果负责。