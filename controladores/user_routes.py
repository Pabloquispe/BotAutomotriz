from flask import Blueprint, render_template, session, redirect, url_for, request, flash
from modelos.models import db, Usuario, Vehiculo, Reserva
from .decorators import login_required

user_bp = Blueprint('user', __name__, url_prefix='/user')

@user_bp.route('/profile')
@login_required
def perfil():
    usuario = Usuario.query.get(session['user_id'])
    return render_template('user/profile.html', usuario=usuario)

@user_bp.route('/registrar_vehiculo', methods=['GET', 'POST'])
@login_required
def registrar_vehiculo():
    if request.method == 'POST':
        usuario_id = session['user_id']
        marca = request.form['marca']
        modelo = request.form['modelo']
        año = request.form['año']
        
        nuevo_vehiculo = Vehiculo(usuario_id=usuario_id, marca=marca, modelo=modelo, año=año)
        db.session.add(nuevo_vehiculo)
        db.session.commit()
        
        flash('Vehículo registrado con éxito.', 'success')
        return redirect(url_for('user.perfil'))
    
    return render_template('user/registrar_vehiculo.html')

@user_bp.route('/reservas')
@login_required
def listar_reservas():
    usuario_id = session['user_id']
    reservas = Reserva.query.filter_by(usuario_id=usuario_id).all()
    return render_template('user/reservas.html', reservas=reservas)

@user_bp.route('/reserva/nueva', methods=['GET', 'POST'])
@login_required
def nueva_reserva():
    usuario_id = session['user_id']
    vehiculos = Vehiculo.query.filter_by(usuario_id=usuario_id).all()
    if not vehiculos:
        flash('Debe registrar un vehículo antes de hacer una reserva.', 'error')
        return redirect(url_for('user.registrar_vehiculo'))
    
    if request.method == 'POST':
        vehiculo_id = request.form['vehiculo_id']
        servicio_id = request.form['servicio_id']
        slot_id = request.form['slot_id']
        problema = request.form['problema']
        fecha_hora = request.form['fecha_hora']
        
        nueva_reserva = Reserva(usuario_id=usuario_id, vehiculo_id=vehiculo_id, servicio_id=servicio_id, slot_id=slot_id, problema=problema, fecha_hora=fecha_hora)
        db.session.add(nueva_reserva)
        db.session.commit()
        
        flash('Reserva creada con éxito.', 'success')
        return redirect(url_for('user.listar_reservas'))
    
    return render_template('user/nueva_reserva.html', vehiculos=vehiculos)
