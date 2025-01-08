# -*- coding: utf-8 -*-
"""
Gradio 介绍与示例代码 🚀

Gradio 是一个用于快速构建机器学习模型用户界面（UI）的 Python 库。
它允许开发者通过几行代码创建一个交互式的 Web 应用，用户可以通过浏览器与模型进行交互。

功能：
- 支持多种输入输出类型：文本、图像、音频、视频等。
- 实时交互：用户输入后，模型会立即返回结果。
- 易于分享：可以通过链接或 Hugging Face Spaces 分享你的应用。

作者：你的名字
日期：2023-10-10
"""

from transformers import pipeline
import gradio as gr

# 示例 1：文本分类器 📝
def text_classifier_example():
    """
    文本分类器示例：输入一段文本，判断情感倾向 😊😢
    """
    print("🚀 正在加载文本分类模型...")
    classifier = pipeline("text-classification")
    print("✅ 模型加载完成！")

    def classify_text(text):
        result = classifier(text)[0]
        return f"Label: {result['label']}, Score: {result['score']:.2f}"

    # 创建 Gradio 界面
    iface = gr.Interface(
        fn=classify_text,
        inputs="text",
        outputs="text",
        title="📝 文本分类器",
        description="输入一段文本，我会判断它的情感倾向！😊😢",
        examples=[
            ["I love this product! It's amazing!"],
            ["I'm so sad today, everything went wrong."],
            ["The weather is nice, but I'm stuck at work."]
        ]
    )
    return iface

# 示例 2：图像分类器 🖼️
def image_classifier_example():
    """
    图像分类器示例：上传一张图片，识别它的内容 🐱🐶
    """
    print("🚀 正在加载图像分类模型...")
    classifier = pipeline("image-classification")
    print("✅ 模型加载完成！")

    def classify_image(image):
        result = classifier(image)
        return {item['label']: item['score'] for item in result}

    # 创建 Gradio 界面
    iface = gr.Interface(
        fn=classify_image,
        inputs="image",
        outputs="label",
        title="🖼️ 图像分类器",
        description="上传一张图片，我会识别它的内容！🐱🐶"
    )
    return iface

# 示例 3：量化问答机器人 🤖
def quant_ai_bot_example():
    """
    量化问答机器人示例：输入问题和上下文，获取答案 📊
    """
    print("🚀 正在加载量化问答模型...")
    qa_pipeline = pipeline("question-answering", model="bert-base-uncased")
    print("✅ 模型加载完成！")

    def answer_question(question, context):
        result = qa_pipeline(question=question, context=context)
        return result['answer']

    # 创建 Gradio 界面
    iface = gr.Interface(
        fn=answer_question,
        inputs=["text", "text"],
        outputs="text",
        title="📊 Quant AI 问答机器人",
        description="请输入你的量化相关问题，我会尽力回答！🤖",
        examples=[
            ["What is the capital of France?", "France is a country in Europe. The capital of France is Paris."],
            ["What is Black-Scholes model?", "The Black-Scholes model is a mathematical model for pricing options."]
        ]
    )
    return iface

# 主程序入口
if __name__ == "__main__":
    print("🌟 欢迎使用 Gradio 示例程序！")
    print("请选择要运行的示例：")
    print("1. 文本分类器 📝")
    print("2. 图像分类器 🖼️")
    print("3. 量化问答机器人 🤖")

    choice = input("请输入选项编号（1/2/3）：")

    if choice == "1":
        iface = text_classifier_example()
    elif choice == "2":
        iface = image_classifier_example()
    elif choice == "3":
        iface = quant_ai_bot_example()
    else:
        print("❌ 无效选项，请重新运行程序！")
        exit()

    # 启动 Gradio 应用
    print("🖥️ 启动 Gradio 应用...")
    iface.launch()
