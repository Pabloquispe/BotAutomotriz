import os
import re
import requests
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, session
from modelos.models import db, Usuario, Vehiculo, Servicio, Slot, Reserva, RegistroUsuario, RegistroServicio, Interaccion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

# Cargar variables de entorno
load_dotenv()

# Configuración de la API de OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Blueprint para las rutas de conversación
conversacion_bp = Blueprint('conversacion', __name__)

# Función para interactuar con OpenAI
def interactuar_con_openai(consulta):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": consulta}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        return "❌ **Lo siento, hemos superado nuestro límite de solicitudes por ahora. Por favor, intenta de nuevo más tarde.**"
    except openai.error.OpenAIError as e:
        print(f"Error interacting with OpenAI: {e}")
        return "❌ **Ha ocurrido un error al interactuar con OpenAI. Por favor, intenta de nuevo más tarde.**"

# Función para registrar interacciones
def registrar_interaccion(usuario_id, mensaje_usuario, respuesta_bot, es_exitosa):
    nueva_interaccion = Interaccion(
        usuario_id=usuario_id,
        mensaje_usuario=mensaje_usuario,
        respuesta_bot=respuesta_bot,
        es_exitosa=es_exitosa
    )
    db.session.add(nueva_interaccion)
    db.session.commit()

# Función para preprocesar el texto
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)  # eliminar números
    texto = re.sub(r'\s+', ' ', texto)  # eliminar espacios adicionales
    texto = re.sub(r'[^\w\s]', '', texto)  # eliminar caracteres especiales
    return texto

# Función para cargar servicios desde el archivo de texto
def cargar_servicios():
    servicios = {}
    try:
        with open('datos/servicios.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if ':' in line:
                    nombre, descripcion = line.split(':', 1)
                    servicios[preprocesar_texto(nombre.strip())] = preprocesar_texto(descripcion.strip())
                else:
                    print(f"Línea ignorada por formato incorrecto: {line}")
    except FileNotFoundError:
        print("El archivo servicios.txt no fue encontrado.")
    except Exception as e:
        print(f"Error al cargar servicios: {e}")
    return servicios

# Función para cargar problemas y servicios desde el archivo de texto
def cargar_problemas_servicios():
    problemas_servicios = {}
    try:
        with open('datos/problemas.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:  # Ignorar líneas en blanco
                    continue
                if ':' in line:
                    problema, servicio = line.split(':', 1)
                    problemas_servicios[preprocesar_texto(problema.strip())] = preprocesar_texto(servicio.strip())
                else:
                    print(f"Línea ignorada por formato incorrecto: '{line}'")
    except FileNotFoundError:
        print("El archivo problemas.txt no fue encontrado.")
    except Exception as e:
        print(f"Error al cargar problemas y servicios: {e}")
    return problemas_servicios

# Función para encontrar servicio basado en la consulta
def encontrar_servicio(servicios, consulta):
    vectorizer = TfidfVectorizer()
    docs = list(servicios.values())
    tfidf_matrix = vectorizer.fit_transform(docs)
    consulta_vec = vectorizer.transform([preprocesar_texto(consulta)])
    similarities = cosine_similarity(consulta_vec, tfidf_matrix).flatten()
    index = similarities.argmax()
    servicio_principal = list(servicios.keys())[index]
    return servicio_principal, similarities[index]

# Función para encontrar problema basado en la consulta
def encontrar_problema(problemas_servicios, consulta, umbral_similitud=0.2):
    vectorizer = TfidfVectorizer()
    problemas = list(problemas_servicios.keys())
    servicios = list(problemas_servicios.values())

    # Preprocesa los problemas
    problemas_preprocesados = [preprocesar_texto(problema) for problema in problemas]

    tfidf_matrix = vectorizer.fit_transform(problemas_preprocesados)
    consulta_vec = vectorizer.transform([preprocesar_texto(consulta)])
    similarities = cosine_similarity(consulta_vec, tfidf_matrix).flatten()
    
    index = similarities.argmax()
    if similarities[index] >= umbral_similitud:
        problema = problemas[index]
        servicio_recomendado = problemas_servicios[problema]
        return problema, servicio_recomendado, similarities[index]
    else:
        return None, None, 0

# Función para generar slots automáticamente
def generar_slots(servicio_id, fecha_inicio, fecha_fin):
    horario_inicio_manana = datetime.strptime("09:00", '%H:%M').time()
    horario_fin_manana = datetime.strptime("12:00", '%H:%M').time()
    horario_inicio_tarde = datetime.strptime("13:00", '%H:%M').time()
    horario_fin_tarde = datetime.strptime("18:00", '%H:%M').time()

    fecha_inicio = datetime.strptime(fecha_inicio, '%Y-%m-%d').date()
    fecha_fin = datetime.strptime(fecha_fin, '%Y-%m-%d').date()
    delta = timedelta(days=1)

    while fecha_inicio <= fecha_fin:
        current_time = datetime.combine(fecha_inicio, horario_inicio_manana)
        while current_time.time() < horario_fin_manana:
            new_slot = Slot(
                servicio_id=servicio_id,
                fecha=fecha_inicio,
                hora_inicio=current_time.time(),
                hora_fin=(datetime.combine(fecha_inicio, current_time.time()) + timedelta(minutes=60)).time(),
                reservado=False
            )
            db.session.add(new_slot)
            db.session.commit()
            current_time += timedelta(minutes=60)

        current_time = datetime.combine(fecha_inicio, horario_inicio_tarde)
        while current_time.time() < horario_fin_tarde:
            new_slot = Slot(
                servicio_id=servicio_id,
                fecha=fecha_inicio,
                hora_inicio=current_time.time(),
                hora_fin=(datetime.combine(fecha_inicio, current_time.time()) + timedelta(minutes=60)).time(),
                reservado=False
            )
            db.session.add(new_slot)
            db.session.commit()
            current_time += timedelta(minutes=60)

        fecha_inicio += delta

# Ruta para manejar los mensajes del usuario
@conversacion_bp.route('/conversacion', methods=['POST'])
def handle_message():
    servicios = cargar_servicios()
    problemas_servicios = cargar_problemas_servicios()
    
    es_exitosa = False
    UMBRAL_SIMILITUD = 0.2
    message = request.json.get('message')
    
    if 'estado' not in session:
        session['estado'] = 'inicio'
    
    if session['estado'] == "inicio" and message.strip() == '':
        respuesta_bot = "¡Hola! 👋 **Soy tu asistente para la reserva de servicios automotrices.** 🚗 ¿Cómo te puedo ayudar hoy? "
        es_exitosa = True
        registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})
    
    if session['estado'] == "inicio":
        session['estado'] = "solicitar_email"
        session['consultas_iniciadas'] = session.get('consultas_iniciadas', 0) + 1
        session['tiempo_inicio_registro'] = datetime.now()
        respuesta_bot = "Por favor, proporcióname tu correo electrónico. 📧"
        es_exitosa = True
        registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_email":
        email = message.strip()
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            respuesta_bot = "❌ **Por favor, proporciona un correo electrónico válido.**"
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        session['email'] = email
        usuario = Usuario.query.filter_by(email=email).first()
        if usuario:
            session['estado'] = "reservar_servicio"
            session['usuario_id'] = usuario.id
            vehiculo = Vehiculo.query.filter_by(usuario_id=usuario.id).first()
            if vehiculo:
                session['vehiculo_id'] = vehiculo.id
            else:
                respuesta_bot = "**No tienes un vehículo registrado.** 🚗 Por favor, registra tu vehículo primero."
                session['estado'] = "solicitar_marca"
                return jsonify({"message": respuesta_bot})
            respuesta_bot = f"¡Hola de nuevo, **{usuario.nombre}!** 👋 ¿Qué servicio deseas reservar hoy o cuéntame qué problema tiene tu auto?"
            es_exitosa = True
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        else:
            session['estado'] = "solicitar_nombre"
            respuesta_bot = f"**¡Encantado de conocerte!** 😊 Parece que eres un cliente nuevo. Por favor, dime tu nombre completo y apellido."
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_nombre":
        session['nombre_completo'] = message.strip()
        session['estado'] = "solicitar_telefono"
        respuesta_bot = f"Gracias, **{session['nombre_completo']}** 🙏. Ahora, ¿puedes proporcionarme tu número de teléfono? 📞"
        registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_telefono":
        telefono = message.strip()
        if not re.match(r"^\d{9}$", telefono):
            respuesta_bot = "❌ **El número de teléfono debe tener 9 dígitos.** Por favor, proporciona un número de teléfono válido."
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        session['telefono'] = telefono
        session['estado'] = "solicitar_direccion"
        respuesta_bot = f"**Excelente.** 🏡 ¿Cuál es la dirección de tu domicilio?"
        registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_direccion":
        session['direccion'] = message.strip()
        session['estado'] = "solicitar_pais"
        respuesta_bot = f"**Genial.** 🌍 ¿De qué país eres?"
        registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_pais":
        session['pais'] = message.strip()
        session['estado'] = "solicitar_fecha_nacimiento"
        respuesta_bot = f"**Perfecto.** 🎂 ¿Cuál es tu fecha de nacimiento? (formato: AAAA-MM-DD)"
        registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_fecha_nacimiento":
        fecha_nacimiento = message.strip()
        try:
            datetime.strptime(fecha_nacimiento, '%Y-%m-%d')
            session['fecha_nacimiento'] = fecha_nacimiento
            session['estado'] = "solicitar_genero"
            respuesta_bot = f"Gracias. 🙏 ¿Cuál es tu género? (F para Femenino, M para Masculino, Otro)"
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        except ValueError:
            respuesta_bot = "❌ **Formato de fecha incorrecto.** Por favor, proporciona tu fecha de nacimiento en el formato AAAA-MM-DD."
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_genero":
        genero = message.strip().upper()
        if genero in ['F', 'M', 'OTRO']:
            session['genero'] = genero
            session['estado'] = "solicitar_marca"
            respuesta_bot = f"Gracias. 🚗 **¿Cuál es la marca de tu vehículo?**"
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        else:
            respuesta_bot = "❌ **Por favor, elige una opción válida:** F para Femenino, M para Masculino, Otro."
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_marca":
        session['marca'] = message.strip()
        session['estado'] = "solicitar_modelo"
        respuesta_bot = f"**Ok, ahora dime.** 🚗 **¿Cuál es el modelo de tu vehículo?**"
        registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_modelo":
        session['modelo'] = message.strip()
        session['estado'] = "solicitar_año"
        respuesta_bot = f"**Está bien.** 🗓️ **¿Cuál es el año de tu vehículo?**"
        registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_año":
        try:
            session['año'] = int(message.strip())
            if session['año'] > datetime.now().year:
                respuesta_bot = "❌ **El año del vehículo no puede ser en el futuro.** Por favor, proporciona un año válido."
                registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
                return jsonify({"message": respuesta_bot})
            nombre, apellido = session['nombre_completo'].split(" ", 1) if " " in session['nombre_completo'] else (session['nombre_completo'], "")
            session['estado'] = "solicitar_password"
            respuesta_bot = "🔒 **Por favor, proporciona una contraseña para tu cuenta.**"
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        except ValueError:
            respuesta_bot = "❌ **Por favor, proporciona un año válido.**"
            registrar_interaccion(session.get("usuario_id"), message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_password":
        session['password'] = message.strip()
        session['estado'] = "confirmar_password"
        respuesta_bot = "🔒 **Por favor, confirma tu contraseña.**"
        registrar_interaccion(session.get("usuario_id"), '********', respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "confirmar_password":
        session['password_confirmacion'] = message.strip()
        if session['password'] != session['password_confirmacion']:
            session['estado'] = "solicitar_password"
            respuesta_bot = "❌ **Las contraseñas no coinciden.** Por favor, proporciona una contraseña para tu cuenta."
            registrar_interaccion(session.get("usuario_id"), '********', respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

        nombre, apellido = session['nombre_completo'].split(" ", 1) if " " in session['nombre_completo'] else (session['nombre_completo'], "")
        usuario_data = {
            'nombre': nombre,
            'apellido': apellido,
            'email': session['email'],
            'telefono': session['telefono'],
            'direccion': session['direccion'],
            'pais': session['pais'],
            'fecha_nacimiento': session['fecha_nacimiento'],
            'genero': session['genero'],
            'password': session['password'],
            'estado': 'inicio'
        }
        response_usuario = requests.post('http://127.0.0.1:5000/usuarios', json=usuario_data)

        if response_usuario.status_code == 200:
            session['usuario_id'] = response_usuario.json()['usuario']
            vehiculo_data = {
                'usuario_id': session['usuario_id'],
                'marca': session['marca'],
                'modelo': session['modelo'],
                'año': session['año']
            }
            response_vehiculo = requests.post('http://127.0.0.1:5000/vehiculos', json=vehiculo_data)
            if response_vehiculo.status_code == 200:
                session['vehiculo_id'] = response_vehiculo.json()['vehiculo']
                session['estado'] = "reservar_servicio"
                tiempo_fin_registro = datetime.now()
                nuevo_registro = RegistroUsuario(
                    usuario_id=session['usuario_id'],
                    tiempo_inicio=session['tiempo_inicio_registro'],
                    tiempo_fin=tiempo_fin_registro
                )
                db.session.add(nuevo_registro)
                db.session.commit()
                respuesta_bot = f"**Muchas gracias {session['nombre_completo']}** 🙌. **Hemos registrado tu información. Cuéntame,** **¿Qué servicio deseas reservar hoy o cuéntame qué problema tiene tu auto?** 🚗"
                es_exitosa = True
                registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
                return jsonify({"message": respuesta_bot})
            else:
                respuesta_bot = "❌ **Hubo un error al registrar tu vehículo.** Por favor, intenta de nuevo."
                registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
                return jsonify({"message": respuesta_bot})
        else:
            respuesta_bot = "❌ **Hubo un error al registrar tu información.** Por favor, intenta de nuevo."
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

    elif session['estado'] == "reservar_servicio":
        consulta = message.strip().lower()
        session['problema'] = consulta
        session['tiempo_inicio_servicio'] = datetime.now()

        problema, servicio_recomendado, similitud_problema = encontrar_problema(problemas_servicios, consulta)
        servicio_principal, similitud_servicio = encontrar_servicio(servicios, consulta)
        
        if similitud_problema > similitud_servicio:
            servicio = Servicio.query.filter_by(nombre=servicio_recomendado).first()
            if servicio:
                session['servicio_principal'] = servicio.nombre
                session['servicio_id'] = servicio.id
                session['servicio_precio'] = servicio.precio
                respuesta_bot = f"**Posible problema puede ser** '{servicio.nombre}' 🔧. **¿Deseas reservar este servicio,🛠️ otro servicio,💰 consultar precio o tienes una CONSULTA ESPECIFICA de servicios o problemas automotrices?** 🚗"
            else:
                respuesta_bot = "❌ **El servicio que has solicitado no está disponible.** Por favor, elige otro servicio."
        elif similitud_servicio >= UMBRAL_SIMILITUD:
            servicio = Servicio.query.filter_by(nombre=servicio_principal).first()
            if servicio:
                session['servicio_principal'] = servicio_principal
                session['servicio_id'] = servicio.id
                session['servicio_precio'] = servicio.precio
                respuesta_bot = f"**Sí, tenemos el servicio de** '{servicio_principal}' 🔧. **¿Deseas reservar este servicio, 🛠️ otro servicio,💰 consultar precio o tienes una CONSULTA ESPECIFICA de servicios o problemas automotrices?** 🚗"
            else:
                respuesta_bot = "❌ **El servicio que has solicitado no está disponible.** Por favor, elige otro servicio."
        else:
            respuesta_bot = "❌ **El servicio que has solicitado no está disponible.** Por favor, elige otro servicio."

        registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
        session['estado'] = "confirmar_servicio"
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "confirmar_servicio":
        confirmacion = message.strip().lower()
        if "cuanto cuesta" in confirmacion or "costo" in confirmacion or "precio" in confirmacion:
            respuesta_bot = f"💰 **El servicio** '{session['servicio_principal']}' **tiene un costo de** {session['servicio_precio']} **soles. ¿Deseas reservar este servicio, 🛠️ otro servicio 🔍 o tienes una CONSULTA ESPECIFICA de servicios o problemas automotrices?**"
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        elif confirmacion in ['si', 'ok', 'por supuesto', 'sí', 'sí.', 'si.', 'esta bien', 'deseo proceder con la reserva de servicio', 'claro', 'reservar', 'procedo con la reserva', 'claro', 'reservar servicio', 'deseo reservar servicio']:
            session['estado'] = "solicitar_fecha"
            respuesta_bot = "📅 **Por favor, proporciona la fecha para tu reserva (AAAA-MM-DD).**"
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        elif "reservar otro servicio" in confirmacion:
            session['estado'] = "reservar_servicio"
            respuesta_bot = "🛠️ **¿Cuál es el otro servicio que deseas reservar?**"
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        elif "consulta especifica" in confirmacion:
            session['estado'] = "interactuar_con_openai"
            respuesta_bot = "🔍 **¿Puedes preguntarme con más detalle que deseas saber sobre los servicios o problemas automotriz como 🛠️ **en que consiste un afinamiento** 📝  consultame por favor?**"
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        else:
            respuesta_bot = "❌ **No entiendo tu respuesta. Por favor, elige una opción: reservar el servicio, reservar otro servicio, o 🔍 CONSULTA ESPECIFICA.**"
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

    elif session['estado'] == "interactuar_con_openai":
        consulta = message.strip().lower()
        respuesta_openai = interactuar_con_openai(consulta)
        respuesta_bot = f"ℹ️ {respuesta_openai}. ¿💡Hay algo más que quieras saber o deseas proceder con la reserva del servicio🟢 '{session['servicio_principal']}'?"
        registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
        session['estado'] = "confirmar_servicio"
        return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_fecha":
        try:
            session['fecha_reserva'] = datetime.strptime(message.strip(), '%Y-%m-%d').date()
            slots_disponibles = Slot.query.filter_by(fecha=session['fecha_reserva'], reservado=False).all()
            if not slots_disponibles:
                generar_slots(session['servicio_id'], str(session['fecha_reserva']), str(session['fecha_reserva']))
                slots_disponibles = Slot.query.filter_by(fecha=session['fecha_reserva'], reservado=False).all()
                if not slots_disponibles:
                    respuesta_bot = "❌ **Lo siento, no hay slots disponibles para el servicio en la fecha solicitada.** Por favor, elige otra fecha."
                    registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
                    return jsonify({"message": respuesta_bot})
            horarios_disponibles = [slot.hora_inicio.strftime('%H:%M') for slot in slots_disponibles]
            session['estado'] = "solicitar_hora"
            respuesta_bot = f"🕒 **Para la fecha** {session['fecha_reserva']}, **tenemos estos horarios disponibles:** {', '.join(horarios_disponibles)}. **Por favor, selecciona uno de estos horarios (HH:MM).**"
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        except ValueError:
            respuesta_bot = "❌ **Formato de fecha incorrecto.** Por favor, proporciona la fecha para tu reserva (AAAA-MM-DD)."
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

    elif session['estado'] == "solicitar_hora":
        hora_reserva = message.strip()
        try:
            fecha_hora_reserva = datetime.strptime(f"{session['fecha_reserva']} {hora_reserva}", '%Y-%m-%d %H:%M')
            slot = Slot.query.filter_by(fecha=session['fecha_reserva'], hora_inicio=fecha_hora_reserva.time(), reservado=False).first()
            if not slot:
                respuesta_bot = "❌ **Lo siento, no hay slots disponibles para el servicio en la fecha y hora solicitada.** Por favor, elige otra fecha u hora."
                registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
                return jsonify({"message": respuesta_bot})

            reserva_data = {
                'usuario_id': session['usuario_id'],
                'vehiculo_id': session['vehiculo_id'],
                'servicio_id': session['servicio_id'],
                'slot_id': slot.id,
                'problema': session['problema'],
                'fecha_hora': fecha_hora_reserva.strftime('%Y-%m-%d %H:%M:%S')
            }
            response = requests.post('http://127.0.0.1:5000/reservas', json=reserva_data)

            if response.status_code == 200:
                slot.reservado = True
                db.session.commit()
                tiempo_fin_servicio = datetime.now()
                nuevo_registro_servicio = RegistroServicio(
                    reserva_id=response.json()['reserva'],
                    tiempo_inicio=session['tiempo_inicio_servicio'],
                    tiempo_fin=tiempo_fin_servicio
                )
                db.session.add(nuevo_registro_servicio)
                db.session.commit()
                session['estado'] = "despedida"
                session['solicitudes_atendidas'] = session.get('solicitudes_atendidas', 0) + 1
                session['conversiones_realizadas'] = session.get('conversiones_realizadas', 0) + 1
                servicio_principal = Servicio.query.get(session['servicio_id']).nombre
                codigo_reserva = response.json()['reserva']
                respuesta_bot = f"**Reserva creada exitosamente con código** {codigo_reserva} ✅ **para el servicio** '{servicio_principal}' **el** {fecha_hora_reserva.strftime('%Y-%m-%d a las %H:%M')}. **¿Necesitas algo más?** 😊"
                es_exitosa = True
                registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
                return jsonify({"message": respuesta_bot})
            else:
                respuesta_bot = "❌ **Hubo un error al registrar tu reserva.** Por favor, intenta de nuevo."
                registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
                return jsonify({"message": respuesta_bot})
        except ValueError:
            respuesta_bot = "❌ **Formato de hora incorrecto.** Por favor, proporciona la hora para tu reserva (HH:MM)."
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

    elif session['estado'] == "despedida":
        if message.strip().lower() in ['no', 'ninguna', 'gracias', 'nada', 'nada gracias', 'nada más']:
            respuesta_bot = "**Muchas gracias, no dudes en escribirnos. Estamos para servirte.** 🙌"
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            session.clear()  # Limpiar la sesión para reiniciar la conversación
            return jsonify({"message": respuesta_bot})
        else:
            respuesta_bot = "❓ **Lo siento, no entiendo tu mensaje. ¿Puedes reformularlo?**"
            registrar_interaccion(session['usuario_id'], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})

# Registrar el blueprint
def register_routes(app):
    app.register_blueprint(conversacion_bp)

