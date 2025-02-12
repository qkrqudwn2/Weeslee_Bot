import openai
from flask import Flask, jsonify, render_template, request
import anthropic
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from database.embedding_chroma import rag_llama

path='/home/weeslee/chatbot_project/chatbot/mysite'
if path not in sys.path:
    sys.path.insert(0, path)

# API key 설정
openai.api_key = "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"
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)

# 챗봇 엔진 서버 정보
host = "127.0.0.1"      # 챗봇 엔진 서버 IP
port = 8084             # 챗봇 엔진 port

# Flask 애플리케이션
app = Flask(__name__)

# llama 모델 설정
model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# GPT 답변 통신
@app.route('/GPT', methods=['POST'])
def GPT():
    data = request.json
    question = data.get('question')
    response = basic_G(question)
    
    # Extract the response content correctly
    return jsonify({"answer" : response.choices[0].message.content.strip()})

# Claude 답변 통신
@app.route('/claude', methods=['POST'])
def claude():
    data = request.json
    question = data.get('question')

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
    동
if __name__ == '__main__':
    app.run(host, port)



