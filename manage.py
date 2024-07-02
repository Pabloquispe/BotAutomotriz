import os
import sys
from flask import Flask
from flask_migrate import Migrate, upgrade
from config import config_by_name
from modelos.models import db
from controladores.admin_routes import admin_bp
from controladores.user_routes import user_bp
from controladores.auth_routes import auth_bp
from controladores.main_routes import main_bp

def create_app(config_name):
    """Crea y configura la aplicación Flask."""
    app = Flask(__name__, template_folder='vistas/templates', static_folder='vistas/static')
    app.config.from_object(config_by_name[config_name])
    
    # Inicializar la base de datos
    db.init_app(app)
    migrate = Migrate(app, db)

    # Registrar Blueprints
    app.register_blueprint(admin_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    
    with app.app_context():
        from controladores.routes import register_routes
        register_routes(app)
        db.create_all()

    return app

if __name__ == "__main__":
    config_name = os.getenv('FLASK_CONFIG', 'default')
    app = create_app(config_name)

    if len(sys.argv) > 1 and sys.argv[1] == 'db':
        with app.app_context():
            upgrade()
    else:
        app.run(debug=(config_name == 'dev'))
