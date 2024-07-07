from flask import Flask
from flask_session import Session
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')
    
    # Inicializar la sesión
    Session(app)
    
    @app.route('/')
    def index():
        return "Hello, World!"
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)


