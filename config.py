import redis

class Config:
    SECRET_KEY = 'your_secret_key'  # Asegúrate de tener una clave secreta configurada
    SESSION_TYPE = 'redis'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = 'flask_session:'
    SESSION_REDIS = redis.from_url('redis://localhost:6379/0')  # Cambia esto si tu URL de Redis es diferente
