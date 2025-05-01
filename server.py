import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from main import tester

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
  def _set_headers(self, code=200):
    self.send_response(code)
    self.send_header('Content-type', 'application/json')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
    self.end_headers()

  def do_OPTIONS(self):
    self._set_headers(200)

  def do_POST(self):
    if self.path != '/predict': return self.send_error(404, 'Not found')

    content_length = int(self.headers.get('Content-Length', 0))
    if content_length != 784:
      self._set_headers(400)
      self.wfile.write(json.dumps({'error': f'Invalid size, expected 784, got {content_length}'}).encode())
      return

    try:
      raw_data = self.rfile.read(content_length)
      pixel_values = list(raw_data)
      predictions = tester.predict(np.array(pixel_values).reshape(28, 28).astype(np.float64))

      self._set_headers(200)
      self.wfile.write(json.dumps({
        'percentages': list(predictions),
        'prediction': int(np.argmax(predictions)),
      }).encode())

    except Exception as e:
      print(f"Prediction error: {e}")
      self._set_headers(500)
      self.wfile.write(json.dumps({'error': 'An internal server error occurred during prediction'}).encode())

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=5000):
  server_address = ('', port)
  httpd = server_class(server_address, handler_class)
  print(f'Serving on http://localhost:{port}')
  httpd.serve_forever()

if __name__ == '__main__':
  run()

