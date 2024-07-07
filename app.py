from flask import Flask, session
from flask_session import Session
from redis import Redis
import os

app = Flask(__name__)

# Configuración de la sesión
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'flask_session:'
app.config['SESSION_REDIS'] = Redis.from_url(os.environ.get('REDIS_URL'))

# Inicialización de la sesión
Session(app)

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == "__main__":
    app.run()
