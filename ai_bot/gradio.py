# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import gradio as gr

class QuantAIBot:
    def __init__(self, model_name="bert-base-uncased"):
        """
        初始化 Quant AI 机器人，加载模型和分词器 🚀
        """
        print("🚀 正在加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        print("✅ 模型加载完成！")

    def answer_question(self, question, context):
        """
        回答用户的问题 🤖
        """
        print(f"🤖 用户问题: {question}")
        print(f"📚 上下文: {context}")
        result = self.qa_pipeline(question=question, context=context)
        print(f"🎉 回答: {result['answer']}")
        return result['answer']

def create_gradio_interface(bot):
    """
    创建 Gradio 界面 🖥️
    """
    def gradio_answer_question(question, context):
        # 调用 QuantAIBot 的问答功能
        return bot.answer_question(question, context)

    # 定义 Gradio 界面
    iface = gr.Interface(
        fn=gradio_answer_question,  # 推理函数
        inputs=["text", "text"],    # 输入：问题和上下文
        outputs="text",             # 输出：答案
        title="📊 Quant AI 问答机器人",  # 界面标题
        description="请输入你的量化相关问题，我会尽力回答！🤖",  # 界面描述
        examples=[
            ["What is the capital of France?", "France is a country in Europe. The capital of France is Paris."],
            ["What is Black-Scholes model?", "The Black-Scholes model is a mathematical model for pricing options."]
        ]
    )
    return iface

# 主程序入口
if __name__ == "__main__":
    # 初始化 Quant AI 机器人
    bot = QuantAIBot(model_name="bert-base-uncased")  # 可以使用其他模型，如 "bert-base-chinese"

    # 创建 Gradio 界面
    iface = create_gradio_interface(bot)

    # 启动 Gradio 应用
    print("🖥️ 启动 Gradio 应用...")
    iface.launch()
