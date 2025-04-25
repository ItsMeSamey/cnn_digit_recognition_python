from flask import Flask, request, jsonify, make_response
import numpy as np
from train_test import tester

app = Flask(__name__)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
  response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
  return response

@app.route('/predict', methods=['OPTIONS'])
def predict_options():
  return make_response()

@app.route('/predict', methods=['POST'])
def predict_post():
  if not request.data: return jsonify({'error': 'No image data received'}), 400
  if len(request.data) != 28*28: return jsonify({'error': f'Invalid size, expected {28*28}, got {len(request.data)}'}), 400

  try:
    predictions = tester.predict(np.frombuffer(request.data, dtype=np.uint8).reshape(28,28))
    return jsonify({'percentages': predictions.tolist(), 'prediction': int(np.argmax(predictions))})
  except Exception as e:
    print(f"An error occurred during prediction: {e}")
    return jsonify({'error': 'An internal server error occurred during prediction'}), 500

PORT = 5000
if __name__ == '__main__':
  print(f"Starting Flask server on http://localhost:{PORT}")
  app.run(debug=True, port=PORT)

