import openai
import requests
import re
import os
from datetime import datetime, timedelta
from modelos.models import db, Usuario, Vehiculo, Servicio, Slot, Reserva, RegistroUsuario, RegistroServicio, Interaccion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Blueprint, request, jsonify
from openai.error import OpenAIError

# Configuración de la API de OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Variable global para almacenar el estado de la conversación
conversation_state = {
    "usuario_id": None,
    "vehiculo_id": None,
    "nombre_completo": None,
    "email": None,
    "telefono": None,
    "direccion": None,
    "pais": None,
    "fecha_nacimiento": None,
    "genero": None,
    "problema": None,
    "servicio_id": None,
    "fecha_reserva": None,
    "estado": "inicio",
    "consultas_iniciadas": 0,
    "solicitudes_atendidas": 0,
    "conversiones_realizadas": 0,
    "servicio_principal": None,
    "servicio_precio": None,
    "tiempo_inicio_registro": None,
    "tiempo_inicio_servicio": None,
    "password": None,
    "password_confirmacion": None
}

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
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError:
        return "❌ **Lo siento, hemos superado nuestro límite de solicitudes por ahora. Por favor, intenta de nuevo más tarde.**"
    except OpenAIError as e:
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
                if ':' in line:
                    nombre, descripcion = line.strip().split(':', 1)
                    servicios[preprocesar_texto(nombre.strip())] = preprocesar_texto(descripcion.strip())
                else:
                    print(f"Formato incorrecto en la línea: {line.strip()}")
    except FileNotFoundError:
        print("El archivo servicios.txt no fue encontrado.")
    except Exception as e:
        print(f"Error al cargar servicios: {e}")
    return servicios

# Función para cargar problemas y servicios
def cargar_problemas_servicios():
    problemas_servicios = {}
    try:
        with open('datos/problemas.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().lower()
                if ':' in line:
                    problema, servicio = line.split(':', 1)
                    problemas_servicios[preprocesar_texto(problema.strip())] = preprocesar_texto(servicio.strip())
                else:
                    print(f"Línea ignorada por formato incorrecto: {line}")
    except FileNotFoundError:
        print("El archivo problemas.txt no fue encontrado.")
    except Exception as e:
        print(f"Error al cargar problemas y servicios: {e}")
    return problemas_servicios

# Función para encontrar servicio basado en la consulta
def encontrar_servicio(servicios, consulta):
    try:
        vectorizer = TfidfVectorizer()
        docs = list(servicios.values())
        tfidf_matrix = vectorizer.fit_transform(docs)
        consulta_vec = vectorizer.transform([preprocesar_texto(consulta)])
        similarities = cosine_similarity(consulta_vec, tfidf_matrix).flatten()
        index = similarities.argmax()
        servicio_principal = list(servicios.keys())[index]
        return servicio_principal, similarities[index]
    except Exception as e:
        print(f"Error al encontrar servicio: {e}")
        return None, 0

# Función para encontrar problema basado en la consulta
def encontrar_problema(problemas_servicios, consulta):
    try:
        vectorizer = TfidfVectorizer()
        docs = list(problemas_servicios.keys())
        tfidf_matrix = vectorizer.fit_transform(docs)
        consulta_vec = vectorizer.transform([preprocesar_texto(consulta)])
        similarities = cosine_similarity(consulta_vec, tfidf_matrix).flatten()
        index = similarities.argmax()
        problema = list(problemas_servicios.keys())[index]
        servicio_recomendado = problemas_servicios[problema]
        return problema, servicio_recomendado, similarities[index]
    except Exception as e:
        print(f"Error al encontrar problema: {e}")
        return None, None, 0

@app.route('/conversacion', methods=['POST'])
def conversacion():
    try:
        data = request.get_json()
        consulta = data.get('consulta', '').strip()
        
        # Buscar servicio basado en la consulta
        problema, servicio_recomendado, similitud = encontrar_problema(app.problemas_servicios, consulta)
        
        # Interactuar con OpenAI
        respuesta_openai = interactuar_con_openai(consulta)
        
        # Registrar la interacción
        registrar_interaccion(conversation_state["usuario_id"], consulta, respuesta_openai, True)
        
        return jsonify({
            "problema": problema,
            "servicio_recomendado": servicio_recomendado,
            "similitud": similitud,
            "respuesta_openai": respuesta_openai
        })
    except Exception as e:
        app.logger.error(f"Error en la ruta de conversación: {e}")
        return jsonify({"error": "Ocurrió un error al procesar la consulta"}), 500

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

# Función para manejar los mensajes del usuario
def handle_message(message):
    global conversation_state
    servicios = cargar_servicios()
    problemas_servicios = cargar_problemas_servicios()
    
    es_exitosa = False
    UMBRAL_SIMILITUD = 0.2
    
    if conversation_state["estado"] == "inicio" and message.strip() == '':
        respuesta_bot = "¡Hola! 👋 **Soy tu asistente para la reserva de servicios automotrices.** 🚗 ¿Cómo te puedo ayudar hoy? "
        es_exitosa = True
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})
    
    if conversation_state["estado"] == "inicio":
        conversation_state["estado"] = "solicitar_email"
        conversation_state["tiempo_inicio_registro"] = datetime.now()
        respuesta_bot = "Por favor, proporcióname tu correo electrónico. 📧"
        es_exitosa = True
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        return jsonify({"message": respuesta_bot})

    elif conversation_state["estado"] == "solicitar_email":
        email = message.strip()
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            respuesta_bot = "❌ **Por favor, proporciona un correo electrónico válido.**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        conversation_state["email"] = email
        usuario = Usuario.query.filter_by(email=email).first()
        if usuario:
            conversation_state["estado"] = "reservar_servicio"
            conversation_state["usuario_id"] = usuario.id
            vehiculo = Vehiculo.query.filter_by(usuario_id=usuario.id).first()
            if vehiculo:
                conversation_state["vehiculo_id"] = vehiculo.id
            else:
                respuesta_bot = "**No tienes un vehículo registrado.** 🚗 Por favor, registra tu vehículo primero."
                conversation_state["estado"] = "solicitar_marca"
                return jsonify({"message": respuesta_bot})
            respuesta_bot = f"¡Hola de nuevo, **{usuario.nombre}!** 👋 ¿Qué servicio deseas reservar hoy o cuéntame qué problema tiene tu auto?"
            es_exitosa = True
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        else:
            conversation_state["estado"] = "solicitar_nombre"
            respuesta_bot = f"**¡Encantado de conocerte!** 😊 Parece que eres un cliente nuevo. Por favor, dime tu nombre completo y apellido."
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})


    elif conversation_state["estado"] == "solicitar_nombre":
        conversation_state["nombre_completo"] = message.strip()
        conversation_state["estado"] = "solicitar_telefono"
        respuesta_bot = f"Gracias, **{conversation_state['nombre_completo'] } ** 🙏. Ahora, ¿puedes proporcionarme tu número de teléfono? 📞"
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        return respuesta_bot

    elif conversation_state["estado"] == "solicitar_telefono":
        telefono = message.strip()
        if not re.match(r"^\d{9}$", telefono):
            respuesta_bot = "❌ **El número de teléfono debe tener 9 dígitos.** Por favor, proporciona un número de teléfono válido."
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot
        conversation_state["telefono"] = telefono
        conversation_state["estado"] = "solicitar_direccion"
        respuesta_bot = f"**Excelente.** 🏡 ¿Cuál es la dirección de tu domicilio?"
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        return respuesta_bot

    elif conversation_state["estado"] == "solicitar_direccion":
        conversation_state["direccion"] = message.strip()
        conversation_state["estado"] = "solicitar_pais"
        respuesta_bot = f"**Genial.** 🌍 ¿De qué país eres?"
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        return respuesta_bot

    elif conversation_state["estado"] == "solicitar_pais":
        conversation_state["pais"] = message.strip()
        conversation_state["estado"] = "solicitar_fecha_nacimiento"
        respuesta_bot = f"**Perfecto.** 🎂 ¿Cuál es tu fecha de nacimiento? (formato: AAAA-MM-DD)"
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        return respuesta_bot

    elif conversation_state["estado"] == "solicitar_fecha_nacimiento":
        fecha_nacimiento = message.strip()
        try:
            datetime.strptime(fecha_nacimiento, '%Y-%m-%d')
            conversation_state["fecha_nacimiento"] = fecha_nacimiento
            conversation_state["estado"] = "solicitar_genero"
            respuesta_bot = f"Gracias. 🙏 ¿Cuál es tu género? (F para Femenino, M para Masculino, Otro)"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot
        except ValueError:
            respuesta_bot = "❌ **Formato de fecha incorrecto.** Por favor, proporciona tu fecha de nacimiento en el formato AAAA-MM-DD."
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot

    elif conversation_state["estado"] == "solicitar_genero":
        genero = message.strip().upper()
        if genero in ['F', 'M', 'OTRO']:
            conversation_state["genero"] = genero
            conversation_state["estado"] = "solicitar_marca"
            respuesta_bot = f"Gracias. 🚗 **¿Cuál es la marca de tu vehículo?**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot
        else:
            respuesta_bot = "❌ **Por favor, elige una opción válida:** F para Femenino, M para Masculino, Otro."
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot

    elif conversation_state["estado"] == "solicitar_marca":
        conversation_state["marca"] = message.strip()
        conversation_state["estado"] = "solicitar_modelo"
        respuesta_bot = f"**Ok, ahora dime.** 🚗 **¿Cuál es el modelo de tu vehículo?**"
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        return respuesta_bot

    elif conversation_state["estado"] == "solicitar_modelo":
        conversation_state["modelo"] = message.strip()
        conversation_state["estado"] = "solicitar_año"
        respuesta_bot = f"**Está bien.** 🗓️ **¿Cuál es el año de tu vehículo?**"
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        return respuesta_bot

    elif conversation_state["estado"] == "solicitar_año":
        try:
            conversation_state["año"] = int(message.strip())
            if conversation_state["año"] > datetime.now().year:
                respuesta_bot = "❌ **El año del vehículo no puede ser en el futuro.** Por favor, proporciona un año válido."
                registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
                return respuesta_bot
            nombre, apellido = conversation_state["nombre_completo"].split(" ", 1) if " " in conversation_state["nombre_completo"] else (conversation_state["nombre_completo"], "")
            conversation_state["estado"] = "solicitar_password"
            respuesta_bot = "🔒 **Por favor, proporciona una contraseña para tu cuenta.**"
            registrar_interaccion(conversation_state["usuario_id"], '********', respuesta_bot, es_exitosa)
            return respuesta_bot
        except ValueError:
            respuesta_bot = "❌ **Por favor, proporciona un año válido.**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot

    elif conversation_state["estado"] == "solicitar_password":
        conversation_state["password"] = message.strip()
        conversation_state["estado"] = "confirmar_password"
        respuesta_bot = "🔒 **Por favor, confirma tu contraseña.**"
        registrar_interaccion(conversation_state["usuario_id"], '********', respuesta_bot, es_exitosa)
        return respuesta_bot

    elif conversation_state["estado"] == "confirmar_password":
        conversation_state["password_confirmacion"] = message.strip()
        if conversation_state["password"] != conversation_state["password_confirmacion"]:
            conversation_state["estado"] = "solicitar_password"
            respuesta_bot = "❌ **Las contraseñas no coinciden.** Por favor, proporciona una contraseña para tu cuenta."
            registrar_interaccion(conversation_state["usuario_id"], '********', respuesta_bot, es_exitosa)
            return respuesta_bot

        nombre, apellido = conversation_state["nombre_completo"].split(" ", 1) if " " in conversation_state["nombre_completo"] else (conversation_state["nombre_completo"], "")
        usuario_data = {
            'nombre': nombre,
            'apellido': apellido,
            'email': conversation_state["email"],
            'telefono': conversation_state["telefono"],
            'direccion': conversation_state["direccion"],
            'pais': conversation_state["pais"],
            'fecha_nacimiento': conversation_state["fecha_nacimiento"],
            'genero': conversation_state["genero"],
            'password': conversation_state["password"],
            'estado': 'inicio'
        }
        response_usuario = requests.post('http://127.0.0.1:5000/usuarios', json=usuario_data)

        if response_usuario.status_code == 200:
            conversation_state["usuario_id"] = response_usuario.json()['usuario']
            vehiculo_data = {
                'usuario_id': conversation_state["usuario_id"],
                'marca': conversation_state["marca"],
                'modelo': conversation_state["modelo"],
                'año': conversation_state["año"]
            }
            response_vehiculo = requests.post('http://127.0.0.1:5000/vehiculos', json=vehiculo_data)
            if response_vehiculo.status_code == 200:
                conversation_state["vehiculo_id"] = response_vehiculo.json()['vehiculo']
                conversation_state["estado"] = "reservar_servicio"
                tiempo_fin_registro = datetime.now()
                nuevo_registro = RegistroUsuario(
                    usuario_id=conversation_state["usuario_id"],
                    tiempo_inicio=conversation_state["tiempo_inicio_registro"],
                    tiempo_fin=tiempo_fin_registro
                )
                db.session.add(nuevo_registro)
                db.session.commit()
                respuesta_bot = f"**Muchas gracias {conversation_state['nombre_completo']}** 🙌. **Hemos registrado tu información. Cuéntame,** **¿Qué servicio deseas reservar hoy o cuéntame qué problema tiene tu auto?** 🚗"
                es_exitosa = True
                registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
                return respuesta_bot
            else:
                respuesta_bot = "❌ **Hubo un error al registrar tu vehículo.** Por favor, intenta de nuevo."
                registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
                return respuesta_bot
        else:
            respuesta_bot = "❌ **Hubo un error al registrar tu información.** Por favor, intenta de nuevo."
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot

    elif conversation_state["estado"] == "reservar_servicio":
        consulta = message.strip().lower()
        conversation_state["problema"] = consulta
        conversation_state["tiempo_inicio_servicio"] = datetime.now()

        problema, servicio_recomendado, similitud_problema = encontrar_problema(problemas_servicios, consulta)
        servicio_principal, similitud_servicio = encontrar_servicio(servicios, consulta)
        
        if similitud_problema > similitud_servicio and similitud_problema >= UMBRAL_SIMILITUD:
            servicio = Servicio.query.filter_by(nombre=servicio_recomendado).first()
            if servicio:
                conversation_state["servicio_principal"] = servicio.nombre
                conversation_state["servicio_id"] = servicio.id
                conversation_state["servicio_precio"] = servicio.precio
                respuesta_bot = f"**Posible problema puede ser** '{servicio.nombre}' 🔧. **¿Deseas reservar este servicio,🛠️ otro servicio,💰 consultar precio o tienes una CONSULTA ESPECIFICA de servicios o problemas automotrices?** 🚗"
            else:
                respuesta_bot = "❌ **El servicio que has solicitado no está disponible.** Por favor, elige otro servicio."
        elif similitud_servicio >= UMBRAL_SIMILITUD:
            servicio = Servicio.query.filter_by(nombre=servicio_principal).first()
            if servicio:
                conversation_state["servicio_principal"] = servicio_principal
                conversation_state["servicio_id"] = servicio.id
                conversation_state["servicio_precio"] = servicio.precio
                respuesta_bot = f"**Sí, tenemos el servicio de** '{servicio_principal}' 🔧. **¿Deseas reservar este servicio, 🛠️ otro servicio,💰 consultar precio o tienes una CONSULTA ESPECIFICA de servicios o problemas automotrices?** 🚗"
            else:
                respuesta_bot = "❌ **El servicio que has solicitado no está disponible.** Por favor, elige otro servicio."
        else:
            respuesta_bot = "❌ **El servicio que has solicitado no está disponible.** Por favor, elige otro servicio."

        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        conversation_state["estado"] = "confirmar_servicio"
        return jsonify({"message": respuesta_bot})

    # Continuar con los demás estados aquí...

    elif conversation_state["estado"] == "confirmar_servicio":
        confirmacion = message.strip().lower()
        if "cuanto cuesta" in confirmacion or "costo" in confirmacion or "precio" in confirmacion:
            respuesta_bot = f"💰 **El servicio** '{conversation_state['servicio_principal']}' **tiene un costo de** {conversation_state['servicio_precio']} **soles. ¿Deseas reservar este servicio, 🛠️ otro servicio 🔍 o tienes una CONSULTA ESPECIFICA de servicios o problemas automotrices?**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        elif "reservar" in confirmacion:
            conversation_state["estado"] = "solicitar_fecha"
            respuesta_bot = "📅 **Por favor, proporciona la fecha para tu reserva (AAAA-MM-DD).**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        elif "otro servicio" in confirmacion or "nuevo servicio" in confirmacion:
            conversation_state["estado"] = "reservar_servicio"
            respuesta_bot = "🛠️ **¿Cuál es el otro servicio que deseas reservar?**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        elif "consulta especifica" in confirmacion:
            conversation_state["estado"] = "interactuar_con_openai"
            respuesta_bot = "🔍 **¿Puedes proporcionar más detalles sobre tu consulta específica?**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})
        else:
            respuesta_bot = "❌ **No entiendo tu respuesta. Por favor, elige una opción: reservar el servicio, reservar otro servicio, o 🔍 CONSULTA ESPECIFICA.**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return jsonify({"message": respuesta_bot})


    elif conversation_state["estado"] == "interactuar_con_openai":
        consulta = message.strip().lower()
        respuesta_openai = interactuar_con_openai(consulta)
        respuesta_bot = f"ℹ️ {respuesta_openai}. ¿Hay algo más que quieras saber o deseas proceder con la reserva del servicio '{conversation_state['servicio_principal']}'?"
        registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
        conversation_state["estado"] = "confirmar_servicio"
        return respuesta_bot

    elif conversation_state["estado"] == "solicitar_fecha":
        try:
            conversation_state["fecha_reserva"] = datetime.strptime(message.strip(), '%Y-%m-%d').date()
            slots_disponibles = Slot.query.filter_by(fecha=conversation_state["fecha_reserva"], reservado=False).all()
            if not slots_disponibles:
                generar_slots(conversation_state["servicio_id"], str(conversation_state["fecha_reserva"]), str(conversation_state["fecha_reserva"]))
                slots_disponibles = Slot.query.filter_by(fecha=conversation_state["fecha_reserva"], reservado=False).all()
                if not slots_disponibles:
                    respuesta_bot = "❌ **Lo siento, no hay slots disponibles para el servicio en la fecha solicitada.** Por favor, elige otra fecha."
                    registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
                    return respuesta_bot
            horarios_disponibles = [slot.hora_inicio.strftime('%H:%M') for slot in slots_disponibles]
            conversation_state["estado"] = "solicitar_hora"
            respuesta_bot = f"🕒 **Para la fecha** {conversation_state['fecha_reserva']}, **tenemos estos horarios disponibles:** {', '.join(horarios_disponibles)}. **Por favor, selecciona uno de estos horarios (HH:MM).**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot
        except ValueError:
            respuesta_bot = "❌ **Formato de fecha incorrecto.** Por favor, proporciona la fecha para tu reserva (AAAA-MM-DD)."
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot

    elif conversation_state["estado"] == "solicitar_hora":
        hora_reserva = message.strip()
        try:
            fecha_hora_reserva = datetime.strptime(f"{conversation_state['fecha_reserva']} {hora_reserva}", '%Y-%m-%d %H:%M')
            slot = Slot.query.filter_by(fecha=conversation_state["fecha_reserva"], hora_inicio=fecha_hora_reserva.time(), reservado=False).first()
            if not slot:
                respuesta_bot = "❌ **Lo siento, no hay slots disponibles para el servicio en la fecha y hora solicitada.** Por favor, elige otra fecha u hora."
                registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
                return respuesta_bot

            reserva_data = {
                'usuario_id': conversation_state["usuario_id"],
                'vehiculo_id': conversation_state["vehiculo_id"],
                'servicio_id': conversation_state["servicio_id"],
                'slot_id': slot.id,
                'problema': conversation_state["problema"],
                'fecha_hora': fecha_hora_reserva.strftime('%Y-%m-%d %H:%M:%S')
            }
            response = requests.post('http://127.0.0.1:5000/reservas', json=reserva_data)

            if response.status_code == 200:
                slot.reservado = True
                db.session.commit()
                tiempo_fin_servicio = datetime.now()
                nuevo_registro_servicio = RegistroServicio(
                    reserva_id=response.json()['reserva'],
                    tiempo_inicio=conversation_state["tiempo_inicio_servicio"],
                    tiempo_fin=tiempo_fin_servicio
                )
                db.session.add(nuevo_registro_servicio)
                db.session.commit()
                conversation_state["estado"] = "despedida"
                conversation_state["solicitudes_atendidas"] += 1
                conversation_state["conversiones_realizadas"] += 1
                servicio_principal = Servicio.query.get(conversation_state["servicio_id"]).nombre
                codigo_reserva = response.json()['reserva']
                respuesta_bot = f"**Reserva creada exitosamente con código** {codigo_reserva} ✅ **para el servicio** '{servicio_principal}' **el** {fecha_hora_reserva.strftime('%Y-%m-%d a las %H:%M')}. **¿Necesitas algo más?** 😊"
                es_exitosa = True
                registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
                return respuesta_bot
            else:
                respuesta_bot = "❌ **Hubo un error al registrar tu reserva.** Por favor, intenta de nuevo."
                registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
                return respuesta_bot
        except ValueError:
            respuesta_bot = "❌ **Formato de hora incorrecto.** Por favor, proporciona la hora para tu reserva (HH:MM)."
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot

    elif conversation_state["estado"] == "despedida":
        if message.strip().lower() in ['no', 'ninguna', 'gracias', 'nada', 'nada gracias', 'nada más']:
            respuesta_bot = "**Muchas gracias, no dudes en escribirnos. Estamos para servirte.** 🙌"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            conversation_state["estado"] = "inicio"  # Reiniciar la conversación
            return respuesta_bot
        else:
            respuesta_bot = "❓ **Lo siento, no entiendo tu mensaje. ¿Puedes reformularlo?**"
            registrar_interaccion(conversation_state["usuario_id"], message, respuesta_bot, es_exitosa)
            return respuesta_bot

