from flask import Flask
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return {"message": "Hello World from Bank Marketing Prediction MLOps!"}

# Add a health check endpoint for the load balancer
@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # Change port from 5000 to 8000 to match firewall and load balancer config
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)