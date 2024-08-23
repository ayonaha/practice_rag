import pandas as pd
from llama_index import GPTSimpleVectorIndex, Document
from flask import Flask, request, jsonify

app = Flask(__name__)

# CSVファイルの読み込みとインデックス作成
df = pd.read_csv('/app/data/qa_data.csv')
documents = [Document(text=row['answer'], metadata={'question': row['question']}) for index, row in df.iterrows()]
index = GPTSimpleVectorIndex(documents)
index.save_to_disk('/app/data/qa_index.json')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data['question']
    search_results = index.query(question)
    context = search_results[0].text  # 最も関連性の高い回答を取得
    return jsonify({'context': context})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
