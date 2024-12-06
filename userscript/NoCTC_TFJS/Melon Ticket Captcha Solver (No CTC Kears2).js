// ==UserScript==
// @name         Melon Ticket Captcha Solver (No CTC Kears2)
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
    const MODEL_URL = 'https://raw.githubusercontent.com/lstrhsu/CaptchaDL/main/userscript/NoCTC_TFJS/noCTCtfjs_model/model.json';

    // 添加常量定义
    const VOCAB = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';  // 验证码可能包含的字符
    const CAPTCHA_LENGTH = 6;  // 验证码长度固定为6
    const CONFIDENCE_THRESHOLD = 0.8;  // 根据实际情况调整

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

    // 加载模型
    let model;
    try {
        console.log('正在加载模型...');
        model = await tf.loadLayersModel(MODEL_URL);

        // 检查模型是否成功加载
        if (!model || !model.layers || model.layers.length === 0) {
            throw new Error('模型加载不完整');
        }

        console.log('模型加载成功，层数:', model.layers.length);
        console.log('模型结构:', model.summary());

        model.compile({
            optimizer: tf.train.adam(),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    } catch (error) {
        console.error('模型加载失败:', error);
        return;
    }

    // 图像预处理函数
    async function preprocessImage(imgElement) {
        return tf.tidy(() => {
            console.log('开始预处理图像...');

            // 转换为张量
            let tensor = tf.browser.fromPixels(imgElement, 4)
                .toFloat();

            // Alpha通道处理
            const alpha = tensor.slice([0, 0, 3], [-1, -1, 1]).div(255.0);
            const rgb = tensor.slice([0, 0, 0], [-1, -1, 3]).div(255.0);
            const whiteBackground = tf.ones(rgb.shape);

            // 使用相同的混合方式
            tensor = tf.add(
                tf.mul(rgb, alpha),
                tf.mul(whiteBackground, tf.sub(1, alpha))
            );

            // 转换为灰度图
            tensor = tf.image.rgbToGrayscale(tensor);

            // 调整大小到模型期望的尺寸
            tensor = tf.image.resizeBilinear(tensor, [80, 280]);

            // 扩展维度 [1, 80, 280, 1]
            tensor = tensor.expandDims(0);

            return tensor;
        });
    }

    // 修改验证码识别函数
    async function recognizeCaptcha(imgElement) {
        let inputTensor = null;
        let predictions = null;

        try {
            inputTensor = await preprocessImage(imgElement);

            // 获取模型预测
            predictions = model.predict(inputTensor);

            // 解码预测结果
            let result = '';

            // 处理每个字符位置的预测
            for (let i = 0; i < CAPTCHA_LENGTH; i++) {
                const charPrediction = predictions[i];
                const maxProb = tf.max(charPrediction).dataSync()[0];

                if (maxProb < CONFIDENCE_THRESHOLD) {
                    console.warn(`字符${i+1}的预测置信度过低: ${maxProb}`);
                    // 可以选择返回null或继续处理
                }

                const maxIndex = tf.argMax(charPrediction, 1).dataSync()[0];
                result += VOCAB[maxIndex];
            }

            if (predictions.length !== CAPTCHA_LENGTH) {
                console.error(`预测结果长度 ${predictions.length} 与预期长度 ${CAPTCHA_LENGTH} 不符`);
                return null;
            }

            return result;  // 已经是大写，无需toUpperCase()

        } catch (error) {
            console.error("验证码识别出错:", error);
            return null;
        } finally {
            if (inputTensor) tf.dispose(inputTensor);
            if (predictions) {
                if (Array.isArray(predictions)) {
                    predictions.forEach(p => tf.dispose(p));
                } else {
                    tf.dispose(predictions);
                }
            }
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
                        console.log('检测到新的验证码图，开始识别...');
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

            // 直接使用��始图像进行识别
            const captchaText = await recognizeCaptcha(img);
            if (!captchaText) {
                console.error('验证码识别失败');
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
                    console.log('验证码已自动填入并提交:', captchaText);
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
            console.log('正在初始化...');

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