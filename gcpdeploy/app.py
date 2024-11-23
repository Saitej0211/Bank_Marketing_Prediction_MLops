from flask import Flask

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def hello_world():
    """Simple Hello World route"""
    return "Hello World"

@app.route("/health")
def health_check():
    """Health check route"""
    return "ok", 200

if __name__ == "__main__":
    # Run the app on port 8000
    app.run(host="0.0.0.0", port=8000)