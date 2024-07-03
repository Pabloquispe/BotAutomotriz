import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_migrate import Migrate
from config import config_by_name
from modelos.models import db
from controladores.conversacion import conversacion_bp
from flask_session import Session
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def create_app(config_name):
    """Crea y configura la aplicación Flask."""
    app = Flask(__name__, template_folder='vistas/templates', static_folder='vistas/static')
    app.config.from_object(config_by_name[config_name])
    
    # Configurar Flask-Session
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)

    # Verificar si la configuración de la base de datos está correcta
    if 'SQLALCHEMY_DATABASE_URI' not in app.config:
        raise RuntimeError("SQLALCHEMY_DATABASE_URI no está configurado")

    db.init_app(app)
    Migrate(app, db)

    # Registrar Blueprints
    app.register_blueprint(conversacion_bp)

    with app.app_context():
        db.create_all()

    # Configuración de logs
    configure_logging(app)

    return app

def configure_logging(app):
    """Configura los logs de la aplicación."""
    if not app.debug:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('Aplicación iniciada')
