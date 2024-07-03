from flask import request, jsonify, render_template, current_app as app, redirect, url_for
from modelos.models import db, Usuario, Vehiculo, Servicio, Slot, Reserva, ComentarioServicio, Repuesto
from controladores.conversacion import handle_message, registrar_interaccion
import traceback

def register_routes(app):
    @app.route('/')
    def home():
        return redirect(url_for('auth.login'))

    @app.route('/conversacion', methods=['POST'])
    def conversacion():
        try:
            user_message = request.json.get('message')
            if not user_message:
                # Responder con mensaje de bienvenida si el mensaje del usuario está vacío
                respuesta_bot = "¡Hola! 👋 **Soy tu asistente para la reserva de servicios automotrices.** 🚗 ¿Cómo te puedo ayudar hoy? "
                es_exitosa = True
                registrar_interaccion(None, '', respuesta_bot, es_exitosa)
                return jsonify({'message': respuesta_bot})

            bot_response = handle_message(user_message)
            return jsonify({'message': bot_response})
        except Exception as e:
            error_trace = traceback.format_exc()
            app.logger.error(f"Error en la ruta '/conversacion': {str(e)}\n{error_trace}")
            return jsonify({'error': str(e)}), 500

    @app.route('/usuarios', methods=['POST'])
    def create_usuario():
        data = request.get_json()
        try:
            new_usuario = Usuario(
                nombre=data['nombre'],
                apellido=data['apellido'],
                email=data['email'],
                telefono=data['telefono'],
                direccion=data.get('direccion'),
                ciudad=data.get('ciudad'),
                profesion=data.get('profesion'),
                pais=data.get('pais'),
                fecha_nacimiento=data.get('fecha_nacimiento'),
                genero=data.get('genero'),
                preferencias_servicio=data.get('preferencias_servicio'),
                rol=data.get('rol', 'usuario'),
                activo=data.get('activo', True),
                estado=data.get('estado', 'inicio')
            )
            
            if 'password' in data:
                new_usuario.set_password(data['password'])
            db.session.add(new_usuario)
            db.session.commit()
            return jsonify({'message': 'Usuario creado', 'usuario': new_usuario.id})
        except Exception as e:
            db.session.rollback()
            error_trace = traceback.format_exc()
            app.logger.error(f"Error en la ruta '/usuarios': {str(e)}\n{error_trace}")
            return jsonify({'error': str(e)}), 500

    @app.route('/vehiculos', methods=['POST'])
    def create_vehiculo():
        data = request.get_json()
        try:
            new_vehiculo = Vehiculo(
                usuario_id=data['usuario_id'],
                marca=data['marca'],
                modelo=data['modelo'],
                año=data['año']
            )
            db.session.add(new_vehiculo)
            db.session.commit()
            return jsonify({'message': 'Vehículo creado', 'vehiculo': new_vehiculo.id})
        except Exception as e:
            db.session.rollback()
            error_trace = traceback.format_exc()
            app.logger.error(f"Error en la ruta '/vehiculos': {str(e)}\n{error_trace}")
            return jsonify({'error': str(e)}), 500

    @app.route('/servicios', methods=['POST'])
    def create_servicio():
        data = request.get_json()
        try:
            new_servicio = Servicio(
                nombre=data['nombre'],
                descripcion=data.get('descripcion'),
                duracion=data.get('duracion'),
                precio=data.get('precio')
            )
            db.session.add(new_servicio)
            db.session.commit()
            return jsonify({'message': 'Servicio creado', 'servicio': new_servicio.id})
        except Exception as e:
            db.session.rollback()
            error_trace = traceback.format_exc()
            app.logger.error(f"Error en la ruta '/servicios': {str(e)}\n{error_trace}")
            return jsonify({'error': str(e)}), 500

    @app.route('/slots', methods=['POST'])
    def create_slot():
        data = request.get_json()
        try:
            new_slot = Slot(
                servicio_id=data['servicio_id'],
                fecha=data['fecha'],
                hora_inicio=data['hora_inicio'],
                hora_fin=data['hora_fin'],
                reservado=data.get('reservado', False)
            )
            db.session.add(new_slot)
            db.session.commit()
            return jsonify({'message': 'Slot creado', 'slot': new_slot.id})
        except Exception as e:
            db.session.rollback()
            error_trace = traceback.format_exc()
            app.logger.error(f"Error en la ruta '/slots': {str(e)}\n{error_trace}")
            return jsonify({'error': str(e)}), 500

    @app.route('/reservas', methods=['POST'])
    def create_reserva():
        data = request.get_json()
        try:
            if 'vehiculo_id' not in data or data['vehiculo_id'] is None:
                raise ValueError("vehiculo_id no puede ser nulo")
            
            new_reserva = Reserva(
                usuario_id=data['usuario_id'],
                vehiculo_id=data['vehiculo_id'],
                servicio_id=data['servicio_id'],
                slot_id=data['slot_id'],
                problema=data['problema'],
                fecha_hora=data['fecha_hora']
            )
            db.session.add(new_reserva)
            slot = Slot.query.get(data['slot_id'])
            slot.reservado = True
            db.session.commit()
            return jsonify({'message': 'Reserva creada', 'reserva': new_reserva.id})
        except Exception as e:
            db.session.rollback()
            error_trace = traceback.format_exc()
            app.logger.error(f"Error en la ruta '/reservas': {str(e)}\n{error_trace}")
            return jsonify({'error': str(e)}), 500
