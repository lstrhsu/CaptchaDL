// ==UserScript==
// @name         Melon Ticket Captcha Solver (CTC Kears2)
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  自动识别 Melon Ticket 验证码
// @author       Your Name
// @match        https://tkglobal.melon.com/reservation/popup/onestop.htm*
// @require      https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0
// @grant        none
// ==/UserScript==

(async function() {
    'use strict';

    // 模型 URL
    const MODEL_URL = 'https://raw.githubusercontent.com/lstrhsu/CaptchaDL/main/userscript/CTC_TFJS/model_js/model.json';

    // 添加常量定义
    const MAX_LENGTH = 6;  // 验证码长度固定为6
    const CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';  // 验证码可能包含的字符

    // CTC 相关常量
    const DELIMITER = '-';
    const CTC_LOSS_USE_ARRAY_ENGINE = "CTC_LOSS_USE_ARRAY_ENGINE";
    const CTC_LOSS_USE_DY_IN_GRAD_FUNC = "CTC_LOSS_USE_DY_IN_GRAD_FUNC";

    // 注册和设置环境变量
    tf.env().registerFlag(CTC_LOSS_USE_ARRAY_ENGINE, () => false);
    tf.env().registerFlag(CTC_LOSS_USE_DY_IN_GRAD_FUNC, () => false);
    tf.env().set('CTC_LOSS_USE_ARRAY_ENGINE', true);

    // 首先定义 CTC 实现
    const ctcImplementation = {
        forwardProb: function(logits, labels, labelLengths, logitLengths) {
            return tf.tidy(() => {
                const batchSize = logits.shape[0];
                const maxTime = logits.shape[1];
                const numClasses = logits.shape[2];

                let alpha = tf.zeros([batchSize, maxTime, labelLengths * 2 + 1]);
                alpha = alpha.scatter(tf.tensor1d([0]), logits.slice([0, 0, labels[0]]));
                alpha = alpha.scatter(tf.tensor1d([1]), logits.slice([0, 0, numClasses - 1]));

                for (let t = 1; t < maxTime; t++) {
                    const prev = alpha.slice([0, t-1, 0], [batchSize, 1, labelLengths * 2 + 1]);
                    const curr = logits.slice([0, t, 0], [batchSize, 1, numClasses]);

                    let transitions = tf.zeros([labelLengths * 2 + 1, labelLengths * 2 + 1]);
                    for (let i = 0; i < labelLengths * 2; i++) {
                        transitions = transitions.scatter(
                            tf.tensor2d([[i, i], [i, i+1]]),
                            tf.ones([2])
                        );
                    }

                    alpha = alpha.slice([0, t, 0]).add(
                        tf.matMul(prev, transitions).mul(curr)
                    );
                }

                return alpha;
            });
        },

        ctcLoss: function(labels, logits) {
            return tf.tidy(() => {
                const labelLengths = labels.shape[1];
                const logitLengths = logits.shape[1];

                const alpha = this.forwardProb(logits, labels, labelLengths, logitLengths);
                const lastTime = logitLengths - 1;
                const lastLabel = labelLengths * 2;
                const finalProb = alpha.slice([0, lastTime, lastLabel]);

                return tf.neg(tf.log(finalProb));
            });
        },

        prepareTensors: function(labels, predictions, delimiterIndex) {
            return tf.tidy(() => {
                console.log('准备 CTC 张量...');
                const paddedInput = predictions.pad([[0, 0], [0, 0], [0, 1]]);
                console.log('填充后的输入形状:', paddedInput.shape);

                // 解码 one-hot 标签
                const batchDecodedLabels = labels.argMax(-1);
                console.log('解码后的标签形状:', batchDecodedLabels.shape);

                // 处理标签序列
                const labelArray = batchDecodedLabels.arraySync();
                const processedLabels = labelArray.map(x => {
                    const ret = [];
                    x.filter(y => y !== delimiterIndex).forEach(z => {
                        ret.push(delimiterIndex, z);
                    });
                    ret.push(delimiterIndex);
                    return ret;
                });

                console.log('处理后的标签序列:', processedLabels);

                // 创建扩展标签张量
                const batchPaddedExtendedLabels = tf.tensor(processedLabels);
                const paddedBatchY = tf.gather(paddedInput, batchPaddedExtendedLabels, 2);
                const batchExtendedLabelLengths = tf.tensor(processedLabels.map(x => x.length));

                return {
                    batchPaddedExtendedLabels,
                    batchPaddedY: paddedBatchY,
                    batchExtendedLabelLengths
                };
            });
        },

        decode: function(logits) {
            return tf.tidy(() => {
                console.log('开始 CTC 解码...');
                console.log('输入 logits 形状:', logits.shape);

                // 修改标签列表构建方式
                const labelsPlusDelimiter = [...CHARACTERS];
                console.log('标签列表(不含分隔符):', labelsPlusDelimiter);

                // 使用 softmax 获取概率，但排除最后一个分隔符位置
                const probs = tf.softmax(logits.slice([0, 0, 0], [-1, -1, CHARACTERS.length]));
                console.log('softmax 后的形状:', probs.shape);

                const bestPath = tf.argMax(probs, -1);
                const path = bestPath.arraySync();
                console.log('解码前的原始路径:', path);

                // CTC 解码
                let result = [];
                let prev = null;

                for (const sequence of path) {
                    let current = '';
                    for (let i = 0; i < sequence.length; i++) {
                        const symbol = sequence[i];
                        if (symbol !== prev) {
                            current += CHARACTERS[symbol];
                            console.log(`时间步 ${i}: 添加字符 ${CHARACTERS[symbol]}`);
                        }
                        prev = symbol;
                    }
                    if (current.length > 0) {
                        result.push(current);
                    }
                }

                let finalResult = result.join('').slice(0, MAX_LENGTH);
                console.log('最终解码结果:', finalResult);
                return finalResult;
            });
        },

        validateSequence: function(sequence) {
            return sequence.length === MAX_LENGTH &&
                   sequence.split('').every(char => CHARACTERS.includes(char));
        },

        postProcessResult: function(result) {
            // 移除无效字符和分隔符
            result = result.replace(new RegExp(`[^${CHARACTERS}]`, 'g'), '');

            // 确保长度正确
            if (result.length > MAX_LENGTH) {
                result = result.slice(0, MAX_LENGTH);
            } else while (result.length < MAX_LENGTH) {
                result += CHARACTERS[0];
            }

            return result;
        },

        forwardTensor: function(batchPaddedExtendedLabels, batchPaddedY, batchExtendedLabelLengths, sequenceLength, labelPadder) {
            if (tf.env().getBool(CTC_LOSS_USE_ARRAY_ENGINE)) {
                // 使用数组方法
                const fwd = this.forwardArray(
                    batchPaddedExtendedLabels.arraySync(),
                    batchPaddedY.transpose([0, 2, 1]).arraySync(),
                    sequenceLength,
                    labelPadder
                );
                return {
                    batchPaddedAlpha: tf.tensor(fwd.batchAlpha),
                    batchLoss: tf.tensor(fwd.batchLoss)
                };
            }

            // 使用张方法
            return tf.tidy(() => {
                const shiftedBpel = batchPaddedExtendedLabels.pad([[0, 0], [2, 0]], -1)
                    .slice(0, batchPaddedExtendedLabels.shape);
                const summaryChooser = batchPaddedExtendedLabels.equal(shiftedBpel).expandDims(1);
                const padddingMask = batchPaddedExtendedLabels.notEqual(tf.scalar(labelPadder)).expandDims(1);

                const bpysDim1 = batchPaddedY.shape[1];
                const bpysDim2 = batchPaddedY.shape[2];

                // 初始化
                const init = batchPaddedY.slice([0, 0], [batchPaddedY.shape[0], 1, 2])
                    .pad([[0, 0], [0, 0], [0, bpysDim2-2]]);
                const c0 = init.sum(2, true);

                let prevStep = init.divNoNan(c0);
                let loss = c0.log().neg();

                const stackable = [prevStep];

                // 前向计算
                for (let i = 1; i < bpysDim1; i++) {
                    const y = batchPaddedY.slice([0, i, 0], [batchPaddedY.shape[0], 1, bpysDim2]);
                    const fwdMask = this.prepareFwdMask(batchExtendedLabelLengths, batchPaddedY.shape, i);

                    const rollingSum1 = tf.add(prevStep,
                        prevStep.pad([[0, 0], [0, 0], [1, 0]], 0)
                            .slice([0, 0, 0], prevStep.shape));
                    const rollingSum2 = tf.add(rollingSum1,
                        prevStep.pad([[0, 0], [0, 0], [2, 0]], 0)
                            .slice([0, 0, 0], prevStep.shape));

                    const fwdStep = tf.where(summaryChooser, rollingSum1, rollingSum2)
                        .mul(y).mul(fwdMask).mul(padddingMask);
                    const c = fwdStep.sum(2, true);

                    loss = loss.sub(c.log());
                    prevStep = fwdStep.divNoNan(c);
                    stackable.push(prevStep);
                }

                return {
                    batchPaddedAlpha: tf.stack(stackable, 2).squeeze([1]),
                    batchLoss: loss.squeeze([1, 2])
                };
            });
        },

        prepareFwdMask: function(batchExtendedLabelLengths, bpyShape, timestep) {
            return tf.tidy(() => {
                const batchSize = bpyShape[0];
                const maxLen = bpyShape[2];

                const indices = tf.range(0, maxLen).expandDims(0)
                    .tile([batchSize, 1]);
                const lengths = batchExtendedLabelLengths.expandDims(1)
                    .tile([1, maxLen]);

                return tf.less(indices, lengths.minimum(tf.scalar(timestep * 2 + 2)))
                    .expandDims(1);
            });
        }
    };

    // 定义 L2 正则化器
    class L2 {
        static className = 'L2';

        constructor(config) {
            this.l2 = config?.l2 ?? 0.01;
        }

        apply(weights) {
            return tf.tidy(() => {
                const l2 = tf.scalar(this.l2);
                return tf.mul(l2, tf.sum(tf.square(weights)));
            });
        }

        getConfig() {
            return {
                l2: this.l2
            };
        }

        static fromConfig(cls, config) {
            return new L2(config);
        }
    }

    // 注册 L2 正则化器
    tf.serialization.registerClass(L2);
    console.log('L2 正则化器注册成功');

    // 然后定义 CTCLayer
    class CTCLayer extends tf.layers.Layer {
        static className = 'CTCLayer';

        constructor(config) {
            super(config);
            this.ctcLoss = ctcImplementation.ctcLoss.bind(ctcImplementation);
        }

        call(inputs) {
            const [y_true, y_pred] = inputs;
            return this.ctcLoss(y_true, y_pred);
        }

        getConfig() {
            return super.getConfig();
        }
    }

    // 注册自定义层
    tf.serialization.registerClass(CTCLayer);
    console.log('CTC Layer 注册成功');

    // 加载模型
    let model;
    try {
        console.log('正在加载模型...');
        model = await tf.loadLayersModel(MODEL_URL);

        model.compile({
            optimizer: tf.train.adam(),
            loss: ctcImplementation.ctcLoss.bind(ctcImplementation),
            metrics: ['accuracy']
        });

        console.log('模型加载并配置成功');
    } catch (error) {
        console.error('模型加载失败:', error);
        return;
    }

    // 图像预处理函数
    async function preprocessImage(imgElement) {
        return tf.tidy(() => {
            console.log('开始预处理图像...');
            console.log('原始图像尺寸:', {
                width: imgElement.width,
                height: imgElement.height
            });

            // 直接转换为 RGB 格式（3通道），忽略 Alpha 通道
            let tensor = tf.browser.fromPixels(imgElement, 3).toFloat().div(255.0);
            console.log('转换为张量后的形状:', tensor.shape);

            // 转为灰度、调整大小、反转颜色
            tensor = tf.image.rgbToGrayscale(tensor);
            console.log('灰度化后的形状:', tensor.shape);

            tensor = tf.image.resizeBilinear(tensor, [80, 280]);
            console.log('调整大小后的形状:', tensor.shape);

            tensor = tf.transpose(tf.sub(1, tensor), [1, 0, 2]);
            console.log('转置和反转后的形状:', tensor.shape);

            const finalTensor = tensor.expandDims(0);
            console.log('最终输入张量形状:', finalTensor.shape);

            return finalTensor;
        });
    }

    // 修改验证码识别函数
    async function recognizeCaptcha(imgElement) {
        let inputTensor = null;
        let prediction = null;

        try {
            if (!model || !model.predict) {
                console.error("模型尚未正确加载");
                return null;
            }

            console.log("开始处理验证码图像...");
            inputTensor = await preprocessImage(imgElement);
            console.log("预处理后的输入张量形状:", inputTensor.shape);

            // 使用模型预测
            prediction = model.predict(inputTensor);
            console.log("模型预测输出形状:", prediction.shape);

            // 打印预测的原始值
            const predictionArray = await prediction.array();
            console.log("预测输出的前几个值:", predictionArray[0].slice(0, 5));

            // 使用CTC解码
            const decodedText = ctcImplementation.decode(prediction);
            console.log("原始解码结果:", decodedText);

            // 后处理
            const processedText = ctcImplementation.postProcessResult(decodedText);
            console.log("处理后的结果:", processedText);

            if (!ctcImplementation.validateSequence(processedText)) {
                console.error('验证码格式无效');
                return null;
            }

            return processedText;
        } catch (error) {
            console.error("验证码识别出错:", error);
            console.error("错误堆栈:", error.stack);
            return null;
        } finally {
            // 清理张量
            if (inputTensor) tf.dispose(inputTensor);
            if (prediction) tf.dispose(prediction);
        }
    }

    // 监听验证码图像加载
    function setupCaptchaObserver() {
        // 首先尝试直接处理已存在的验证码图像
        const existingCaptcha = document.querySelector('#captchaImg');
        if (existingCaptcha) {
            console.log('检测到现有验证码图像，开始识别...');
            processCaptchaImage(existingCaptcha);
        }

        // 设置观察器监听新的验证码图像
        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (node.tagName === 'IMG' && node.id === 'captchaImg') {
                        console.log('检测到新的验证码图像，开始识别...');
                        processCaptchaImage(node);
                    }
                }
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    // 处理验证码图像
    async function processCaptchaImage(imgElement) {
        try {
            // 直接获取原始 base64 数据
            const base64Data = imgElement.src;
            console.log('获取到验证码图像数据');

            // 创建新的图像对象用于识别
            const img = new Image();
            img.crossOrigin = 'anonymous';

            await new Promise((resolve, reject) => {
                img.onload = () => {
                    console.log('图像加载完成:', {
                        width: img.width,
                        height: img.height
                    });
                    resolve();
                };
                img.onerror = reject;
                img.src = base64Data;
            });

            // 直接使用原始图像进行识别
            const captchaText = await recognizeCaptcha(img);
            if (!captchaText) {
                console.error('验证码别失败');
                return;
            }

            await fillCaptcha(captchaText);

        } catch (error) {
            console.error('处理验证码时出错:', error);
            console.error('错误堆栈:', error.stack);
        }
    }

    // 修改填入验证码的函数
    async function fillCaptcha(captchaText) {
        if (!captchaText) {
            console.error('验证码文本为空，无法填入');
            return;
        }

        try {
            // 等待输入框出现
            const inputElement = await waitForElement('#label-for-captcha');
            if (inputElement) {
                console.log('找到输入框，准备填入验证码:', captchaText);

                // 清空现有内容
                inputElement.value = '';
                // 填入新验证码
                inputElement.value = captchaText.toUpperCase();
                // 触发输入事件
                inputElement.dispatchEvent(new Event('input', { bubbles: true }));

                // 等待并点击提按钮
                const completeButton = await waitForElement('#btnComplete');
                if (completeButton) {
                    console.log('找到提交按钮，准备点击');
                    completeButton.click();
                    console.log('验码已自动填入并提交:', captchaText);
                }
            }
        } catch (error) {
            console.error('填入验证码时出错:', error);
        }
    }

    // 等待元素出现的辅助函数
    function waitForElement(selector, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const element = document.querySelector(selector);
            if (element) {
                resolve(element);
                return;
            }

            const observer = new MutationObserver((mutations) => {
                const element = document.querySelector(selector);
                if (element) {
                    observer.disconnect();
                    resolve(element);
                }
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true
            });

            // 设置超时
            setTimeout(() => {
                observer.disconnect();
                reject(new Error(`等待元素 ${selector} 超时`));
            }, timeout);
        });
    }

    // 启动脚本
    async function init() {
        try {
            // 设置 CTC 环境变量
            tf.env().set('CTC_LOSS_USE_ARRAY_ENGINE', true);
            console.log('CTC 环境变量设置完成');

            // 加载模型
            console.log('正在加载模型...');
            model = await tf.loadLayersModel(MODEL_URL);

            // 配置型
            model.compile({
                optimizer: tf.train.adam(),
                loss: ctcImplementation.ctcLoss.bind(ctcImplementation),
                metrics: ['accuracy']
            });

            console.log('模型加载并配置成功');
            console.log('模型结构:', model.summary());

            // 启动验证码监听
            setupCaptchaObserver();
        } catch (error) {
            console.error('初始化失败:', error);
            console.error('错误堆栈:', error.stack);
        }
    }

    // 启动脚本
    init();
})();