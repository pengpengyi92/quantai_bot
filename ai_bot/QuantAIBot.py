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

    def run_gradio_app(self):
        """
        启动 Gradio 应用，提供用户界面 🖥️
        """
        print("🖥️ 启动 Gradio 应用...")
        iface = gr.Interface(
            fn=self.answer_question,
            inputs=["text", "text"],
            outputs="text",
            title="📊 Quant AI 机器人",
            description="请输入你的量化相关问题，我会尽力回答！🤖"
        )
        iface.launch()

# 主程序入口
if __name__ == "__main__":
    # 初始化 Quant AI 机器人
    bot = QuantAIBot(model_name="bert-base-uncased")  # 可以使用其他模型，如 "bert-base-chinese"

    # 启动 Gradio 应用
    bot.run_gradio_app()
