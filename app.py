from flask import Flask
from flask_session import Session
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Inicializar la extensión Flask-Session
Session(app)

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == "__main__":
    app.run()
