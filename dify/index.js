const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

app.post('/query', async (req, res) => {
    const question = req.body.question;

    // LlamaIndexにクエリを送信
    const indexResponse = await axios.post('http://llama_index_container:8001/query', { question });
    const context = indexResponse.data.context;

    // LLaMA 3.1に応答生成リクエストを送信
    const llamaResponse = await axios.post('http://llama_model_container:8002/generate', { question, context });
    const answer = llamaResponse.data.answer;

    res.json({ answer });
});

app.listen(3000, () => {
    console.log('Dify service running on port 3000');
});
