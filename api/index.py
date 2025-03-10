import openai
from flask import Flask, jsonify, render_template, request
import anthropic

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys

#from simil import rag_llama3

#from database.embedding_chroma import rag_llama

path='/home/weeslee/chatbot_project/chatbot/mysite'
if path not in sys.path:
    sys.path.insert(0, path)

# Set the API key
openai.api_key = "sk-"
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="sk-",
)


# 챗봇 엔진 서버 정보
host = "127.0.0.1"      # 챗봇 엔진 서버 IP
port = 8084             # 챗봇 엔진 port

# Flask 애플리케이션
app = Flask(__name__)

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

@app.route('/GPT', methods=['POST'])
def GPT():
    data = request.json
    question = data.get('question')
    response = basic_G(question)
    
    # Extract the response content correctly
    return jsonify({"answer" : response.choices[0].message.content.strip()})

@app.route('/claude', methods=['POST'])
def claude():
    data = request.json
    question = data.get('question')

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        temperature=0.6,
        messages=[
        {"role": "user", 
         "content": question}
         ]
         )
    return jsonify({"answer" : response.content[0].text})

@app.route('/llama', methods=['POST'])
def llama():
    data = request.json
    question = data.get('question')
    '''
    messages = [
        {"role": "user", "content": f"{question}"}
       ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    '''
    outputs, input_ids = basic_L(question)
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return jsonify({"answer" : response})

@app.route('/rag', methods=['POST'])
def rag():
    data = request.json
    question = data.get('question')
    '''
    messages = [
        {"role": "user", "content": f"{question}"}
       ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    '''
    answer = rag_llama3(question)
    
    return jsonify({"answer" : answer})

@app.route('/ai3')
def login():
    return render_template('ai_3.html')

@app.route('/')
def home():
    return render_template('rag1.html')

#버튼 상태 값 송수신
@app.route('/get_info', methods=['GET', 'POST'])
def get_info():
    data = request.json
    btns = data.get('btn_state')
    return render_template('main.html', btn=btns)

#프롬프트
def basic_G(question):
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {
                "role": "system", 
             "content": "친절하게 답변해주세요"
             },
             {
                 "role": "user",
                 "content": question
             }
        ],
        max_tokens=512,
        n=1,
        temperature=0.6
    )
    return response

def basic_L(question):
    messages=[
            {
                "role": "system", 
             "content": "같은 말은 반복하지 않고 친절하게게 답변해주세요"
             },
             {
                 "role": "user",
                 "content": question
             }
        ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    return outputs, input_ids


if __name__ == '__main__':
    app.run(host=host, port=port, debug=True)


