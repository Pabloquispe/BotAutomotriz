import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SESSION_TYPE = 'redis'  # O el tipo de sesión que estás utilizando
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = 'flask_session:'
    SESSION_REDIS = redis.from_url(os.environ.get('REDIS_URL'))


