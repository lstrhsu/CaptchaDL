// ==UserScript==
// @name         Melon Ticket Captcha Solver (ONNX Version)
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Captcha recognition using ONNX.js
// @author       Your Name
// @match        https://tkglobal.melon.com/reservation/popup/onestop.htm*
// @require      https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js
// @require      https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js
// @connect      raw.githubusercontent.com
// @connect      cdn.jsdelivr.net
// @grant        GM_xmlhttpRequest
// @grant        unsafeWindow
// @connect      microsoft.github.io
// @resource     WASM_SIMD https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd.wasm
// @resource     WASM https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm.wasm
// ==/UserScript==
(async function() {
    'use strict';

    // 添加常量定义
    const MAX_LENGTH = 6; // 验证码长度
    const CHARACTERS = [
        '', // CTC blank character
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z'
    ];

    console.log('Script execution started...');

    // Initialize ONNX session
    let session;

    // Configure ONNX Runtime
    const initONNX = async () => {
        try {
            // Set WASM paths explicitly to avoid dynamic URL generation
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
            console.log('WASM paths set to:', ort.env.wasm.wasmPaths);

            // Check if ONNX Runtime is loaded correctly
            console.log('ort:', typeof ort);
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded');
            }
            console.log('ONNX Runtime version:', ort.version);

            // Add WebGPU support check
            console.log('Checking WebGPU support...');
            const webGPUSupported = 'gpu' in navigator;
            console.log('WebGPU supported:', webGPUSupported);

            // Configuration options
            const options = {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all',
                //logSeverityLevel: 0,  // 启用详细日志
                executionMode: 'sequential'  // 使用顺序执行模式
            };

            // Use the correct model URL
            const MODEL_URL = 'https://raw.githubusercontent.com/lstrhsu/CaptchaDL/main/userscript/CTC_ONNX/model_original.onnx';
            console.log('Fetching model from:', MODEL_URL);

            // Fetch model data
            console.log('Fetching model...');
            const modelResponse = await new Promise((resolve, reject) => {
                GM_xmlhttpRequest({
                    method: 'GET',
                    url: MODEL_URL,
                    responseType: 'arraybuffer',
                    onload: (response) => {
                        // Validate response
                        if (response.status !== 200) {
                            reject(new Error(`Model download failed: ${response.status}`));
                            return;
                        }
                        resolve(response);
                    },
                    onerror: reject
                });
            });

            // Use ArrayBuffer directly
            const modelBuffer = modelResponse.response;
            console.log('Model data size:', modelBuffer.byteLength, 'bytes');

            // Check if model data is valid
            console.log('Validating model buffer...');
            if (!modelBuffer || modelBuffer.byteLength === 0) {
                throw new Error('Model data is invalid or empty');
            }

            // Create session
            console.log('Creating ONNX Runtime session...');
            console.log('Model buffer size:', modelBuffer.byteLength);
            console.log('Creating session with options:', JSON.stringify(options));

            session = await ort.InferenceSession.create(modelBuffer, options);

            // Validate session
            console.log('Session created successfully');
            console.log('Input nodes:', session.inputNames);
            console.log('Output nodes:', session.outputNames);

            // Log selected execution provider
            console.log('Execution provider used by session:', options.executionProviders[0]);

            // 详细验证session
            if (!session.inputNames || session.inputNames.length === 0) {
                throw new Error('Invalid session: no input names found');
            }

            // 打印详细的模型信息
            console.log('Model inputs:', session.inputNames);
            console.log('Input details:', session.inputNames.map(name => {
                return {
                    name: name,
                    shape: session._model?.inputMetadata?.[name]?.dimensions || [],
                    type: session._model?.inputMetadata?.[name]?.type || 'unknown'
                };
            }));

            return true;
        } catch (error) {
            console.error('详细错误信息:', {
                message: error.message,
                stack: error.stack,
                name: error.name
            });
            return false;
        }
    };

    // Image preprocessing function - keep original logic
    async function preprocessImage(imgElement) {
        return tf.tidy(() => {
            console.log('Starting image preprocessing...');
            console.log('Original image dimensions:', {
                width: imgElement.width,
                height: imgElement.height
            });

            // Convert to RGB format
            let tensor = tf.browser.fromPixels(imgElement, 3).toFloat().div(255.0);
            console.log('Tensor shape after conversion:', tensor.shape);

            // Convert to grayscale
            tensor = tf.image.rgbToGrayscale(tensor);
            console.log('Shape after grayscaling:', tensor.shape);

            // Resize
            tensor = tf.image.resizeBilinear(tensor, [80, 280]);
            console.log('Shape after resizing:', tensor.shape);

            // Invert colors and transpose
            tensor = tf.transpose(tf.sub(1, tensor), [1, 0, 2]);
            console.log('Shape after transpose and inversion:', tensor.shape);

            // Add batch dimension and convert to array
            const finalTensor = tensor.expandDims(0);
            console.log('Final input tensor shape:', finalTensor.shape);

            return finalTensor.arraySync();
        });
    }

    // Decode function
    function decodePrediction(outputTensor) {
        console.log('开始解码预测结果');
        console.log('原始模型输出:', outputTensor);

        // 将Float32Array转换为普通数组并重塑
        const length = outputTensor.length;
        const timeSteps = 70;  // 时间步长
        const numClasses = 27;  // 类别数（包括空白字符）

        // 重塑数组为 [timeSteps, numClasses] 的形式
        const predictions = [];
        for (let t = 0; t < timeSteps; t++) {
            const stepProbs = Array.from(outputTensor.slice(t * numClasses, (t + 1) * numClasses));
            predictions.push(stepProbs);
        }

        let result = '';
        let prev = null;

        // 对每个时间步进行解码
        for (let t = 0; t < predictions.length; t++) {
            const probs = predictions[t];
            // 找出最大概率的索引
            let maxIndex = 0;
            let maxProb = probs[0];
            for (let i = 1; i < numClasses; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }

            // CTC解码：去除重复和空白字符
            if (maxIndex !== prev && maxIndex !== 0) {  // 0是空白字符
                result += CHARACTERS[maxIndex];
            }
            prev = maxIndex;
        }

        // 确保验证码长度正确
        result = result.slice(0, MAX_LENGTH);
        while (result.length < MAX_LENGTH) {
            result += CHARACTERS[1];  // 使用第一个非空字符补充长度
        }

        console.log('解码后的验证码结果:', result);
        return result;
    }

    // Captcha recognition function
    async function recognizeCaptcha(imgElement) {
        try {
            if (!session) {
                console.error("ONNX会话未初始化");
                return null;
            }

            // 预处理图像
            console.log("开始预处理图像...");
            const inputData = await preprocessImage(imgElement);
            console.log("预处理完成，输入数据形状:",
                [inputData.length, inputData[0].length, inputData[0][0].length, inputData[0][0][0].length]);

            // 准备推理输入
            const feeds = {
                [session.inputNames[0]]: new ort.Tensor(
                    'float32',
                    Float32Array.from(inputData.flat(4)),
                    [1, 280, 80, 1]
                )
            };

            // 执行推理
            console.log("开始执行模型推理...");
            const outputData = await session.run(feeds);
            const outputTensor = outputData[session.outputNames[0]];
            console.log("推理完成，输出数据:", outputTensor.data);

            // 解码结果
            const result = decodePrediction(outputTensor.data);
            console.log("验证码识别结果:", result);

            return result;

        } catch (error) {
            console.error("验证码识别过程出错:", error);
            return null;
        }
    }

    // Keep original other functions
    async function processCaptchaImage(imgElement) {
        try {
            const base64Data = imgElement.src;
            console.log('Captcha image data obtained');

            const img = new Image();
            img.crossOrigin = 'anonymous';

            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
                img.src = base64Data;
            });

            const captchaText = await recognizeCaptcha(img);
            if (captchaText) {
                await fillCaptcha(captchaText);
            }
        } catch (error) {
            console.error('Error processing captcha:', error);
        }
    }

    // Keep original fillCaptcha and waitForElement functions...
    async function fillCaptcha(captchaText) {
        if (!captchaText) {
            console.error('验证码文本为空，无法填写');
            return;
        }

        try {
            console.log('准备填写验证码:', captchaText);

            // 等待输入框出现
            const inputElement = await waitForElement('#label-for-captcha');
            if (!inputElement) {
                throw new Error('未找到验证码输入框');
            }

            console.log('已找到验证码输入框');

            // 模拟真实用户输入
            inputElement.focus();
            inputElement.value = captchaText.toUpperCase();

            // 触发必要的输入事件
            inputElement.dispatchEvent(new Event('input', { bubbles: true }));
            inputElement.dispatchEvent(new Event('change', { bubbles: true }));

            console.log('验证码已填入:', captchaText);

            // 等待并点击提交按钮
            const submitButton = await waitForElement('#btnComplete');
            if (submitButton && !submitButton.disabled) {
                console.log('正在点击提交按钮');
                submitButton.click();
            } else {
                console.warn('提交按钮不可用或未找到');
            }

        } catch (error) {
            console.error('填写验证码过程出错:', error);
        }
    }

    function waitForElement(selector, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const element = document.querySelector(selector);
            if (element) return resolve(element);

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

            setTimeout(() => {
                observer.disconnect();
                reject(new Error(`Timeout waiting for element ${selector}`));
            }, timeout);
        });
    }

    // Add captcha observer
    function setupCaptchaObserver() {
        console.log('Setting up captcha observer...');

        // First check if captcha image already exists
        const existingCaptcha = document.querySelector('#captchaImg');
        if (existingCaptcha) {
            console.log('Existing captcha image found, processing...');
            processCaptchaImage(existingCaptcha);
        }

        // Set up MutationObserver to watch for captcha changes
        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                if (mutation.type === 'childList') {
                    const captchaImg = document.querySelector('#captchaImg');
                    if (captchaImg) {
                        console.log('Captcha image change detected, processing...');
                        processCaptchaImage(captchaImg);
                    }
                }
            }
        });

        // Start observing
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        console.log('Captcha observer setup complete');
    }

    // Modify initialization function
    async function init() {
        try {
            console.log('Starting initialization...');

            // Initialize ONNX
            console.log('Initializing ONNX...');
            const success = await initONNX();
            if (!success) {
                throw new Error('ONNX initialization failed');
            }

            console.log('ONNX initialized successfully, setting up captcha observer...');
            // Start captcha observer
            setupCaptchaObserver();

            console.log('Initialization complete');
        } catch (error) {
            console.error('Initialization failed:', error);
            console.error('Error stack:', error.stack);
        }
    }

    // Ensure script runs after page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    console.log('Script loaded');  // Add end log
})();
