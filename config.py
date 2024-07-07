import os
from redis import Redis

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'
    SESSION_TYPE = 'redis'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = 'flask_session:'
    SESSION_REDIS = Redis.from_url(os.environ.get('REDIS_URL') or 'redis://localhost:6379/0')
