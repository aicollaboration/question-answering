import datetime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

API_V1 = '/api/1.0'

@app.route(API_V1 + '/ping', methods=['GET'])
def ping():
    return "pong"

@app.route(API_V1 + '/info', methods=['GET'])
def info():
    return jsonify({
        'version': API_V1,
        'project': '5 elements of AI',
        'service': 'backend',
        'language': 'python',
        'type': 'api',
        'date': str(datetime.datetime.now()),
    })


@app.route(API_V1 + '/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origin='localhost')
def predict():
    return 'no data provided'
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)