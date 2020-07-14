import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

app = Flask(__name__)
cors = CORS(app)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')


API_V1 = '/api/1.0'

@app.route(API_V1 + '/ping', methods=['GET'])
def ping():
    return "pong"

@app.route(API_V1 + '/info', methods=['GET'])
def info():
    return jsonify({
        'version': API_V1,
        'project': '5 elements of AI',
        'service': 'question-answering',
        'language': 'python',
        'type': 'api',
        'date': str(datetime.datetime.now()),
    })


@app.route(API_V1 + '/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origin='localhost')
def predict():
    data = request.json
    
    encoding = tokenizer.encode_plus(data['question'], data['context'])
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    
    print('answer_tokens_to_string ', answer_tokens_to_string)
    
    return jsonify(answer_tokens_to_string)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)