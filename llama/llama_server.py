from flask import Flask, request, jsonify
from transformers import LlamaForCausalLM, LlamaTokenizer

app = Flask(__name__)

# モデルとトークナイザーのロード
model = LlamaForCausalLM.from_pretrained('/app/model')
tokenizer = LlamaTokenizer.from_pretrained('/app/model')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    question = data['question']
    context = data['context']
    
    inputs = tokenizer(f"Question: {question}\nContext: {context}\nAnswer:", return_tensors="pt")
    outputs = model.generate(**inputs)
    
    return jsonify({'answer': tokenizer.decode(outputs[0], skip_special_tokens=True)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)
