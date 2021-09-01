import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import yaml

app = Flask(__name__)
cors = CORS(app)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')


API_V1 = '/api/1.0'

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "it works"
    })

@app.route(API_V1 + '/ping', methods=['GET'])
def ping():
    return "pong"

@app.route(API_V1 + '/definition', methods=['GET'])
def definition():
    with open("./openapi.yml", 'r') as stream:
        try:
            return jsonify(yaml.safe_load(stream))
        except yaml.YAMLError as exception:
            return jsonify(exception)

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
    
    '''
    inputs = tokenizer.encode_plus(data['question'], data['context'], add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    answer_start_scores, answer_end_scores = model(input_ids)
    
    print(answer_start_scores, answer_end_scores)
    
    return jsonify({ 'answer_start_scores': answer_start_scores, 'answer_end_scores': answer_end_scores })
    '''
    
    
    encoding = tokenizer.encode_plus(data['question'], data['context'])
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    
    print('####')
    start = torch.argmax(start_scores)
    end = torch.argmax(end_scores)
    print(start, end)
    print('####')

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    
    print('answer_tokens_to_string ', answer_tokens_to_string)
    
    return jsonify({ 'answer': answer_tokens_to_string })
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)