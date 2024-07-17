from flask import Flask, request, jsonify, session
from flask_session import Session

VALID_KEYWORDS = ['gracias', 'sí', 'no', 'reservar otro servicio']

def initialize_app():
    app = Flask(__name__)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SECRET_KEY'] = 'supersecretkey'
    Session(app)
    return app

app = initialize_app()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip().lower()
    
    if user_input not in VALID_KEYWORDS:
        session.clear()  # Finalizar sesión
        return jsonify({"response": "❌ **El servicio que has solicitado no está disponible.** Por favor, elige 🛠️ Reservar otro servicio."})
    
    # Procesar la entrada válida
    if user_input == 'gracias':
        session.clear()  # Finalizar sesión
        return jsonify({"response": "😊 **¡De nada!** Tu sesión ha sido finalizada. ¡Hasta luego!"})
    
    # Otros casos de uso aquí
    return jsonify({"response": "🔧 **¿En qué más puedo ayudarte?**"})

if __name__ == '__main__':
    app.run(debug=True)

