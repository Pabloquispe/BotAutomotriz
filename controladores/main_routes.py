from flask import Blueprint, redirect, url_for, jsonify

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return redirect(url_for('auth.login'))

@main_bp.route('/api/welcome', methods=['GET'])
def welcome_message():
    respuesta_bot = "¡Hola! 👋 **Soy tu asistente para la reserva de servicios automotrices.** 🚗 ¿Cómo te puedo ayudar hoy? Por favor, proporcióname tu correo electrónico. 📧"
    return jsonify(message=respuesta_bot)
