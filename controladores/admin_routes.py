from flask import Blueprint, request, redirect, url_for, flash, render_template, send_file
from modelos.models import db, Usuario, Vehiculo, Reserva, Servicio
from .decorators import login_required, admin_required
import pandas as pd

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Dashboard del Administrador
@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    total_usuarios = Usuario.query.count()
    total_vehiculos = Vehiculo.query.count()
    total_servicios = Servicio.query.count()
    total_reservas = Reserva.query.count()
    total_no_realizados = Reserva.query.filter_by(estado='no realizado').count()
    return render_template('admin/dashboard.html', 
                           total_usuarios=total_usuarios, 
                           total_vehiculos=total_vehiculos, 
                           total_servicios=total_servicios, 
                           total_reservas=total_reservas,
                           total_no_realizados=total_no_realizados)

# Listar Reservas
@admin_bp.route('/reservas')
@admin_required
def reservas():
    reservas = Reserva.query.all()
    return render_template('admin/reservas.html', reservas=reservas)

# Listar Servicios
@admin_bp.route('/servicios')
@admin_required
def servicios():
    servicios = Servicio.query.all()
    return render_template('admin/servicios.html', servicios=servicios)

# Listar Clientes
@admin_bp.route('/clientes')
@admin_required
def clientes():
    clientes = Usuario.query.all()
    return render_template('admin/clientes.html', clientes=clientes)

# Exportar Clientes a Excel
@admin_bp.route('/exportar_clientes_excel', methods=['GET'])
@admin_required
def exportar_clientes_excel():
    clientes = Usuario.query.all()
    data = {
        "ID": [cliente.id for cliente in clientes],
        "Nombre": [cliente.nombre for cliente in clientes],
        "Apellido": [cliente.apellido for cliente in clientes],
        "Email": [cliente.email for cliente in clientes],
        "Teléfono": [cliente.telefono for cliente in clientes],
        "Dirección": [cliente.direccion for cliente in clientes],
        "Género": [cliente.genero for cliente in clientes],
        "Fecha de Nacimiento": [cliente.fecha_nacimiento for cliente in clientes],
        "País": [cliente.pais for cliente in clientes],
        "Fecha de Registro": [cliente.fecha_registro for cliente in clientes]
    }
    df = pd.DataFrame(data)
    file_path = 'clientes.xlsx'
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)

# Listar Usuarios y Cambiar Roles
@admin_bp.route('/roles')
@admin_required
def roles():
    usuarios = Usuario.query.all()
    return render_template('admin/roles.html', usuarios=usuarios)

@admin_bp.route('/roles/cambiar/<int:user_id>', methods=['POST'])
@admin_required
def cambiar_rol(user_id):
    nuevo_rol = request.form['nuevo_rol']
    user = Usuario.query.get(user_id)
    if user:
        user.rol = nuevo_rol
        db.session.commit()
        flash('Rol actualizado correctamente.', 'success')
    else:
        flash('Usuario no encontrado.', 'danger')
    return redirect(url_for('admin.roles'))

# Crear Nuevo Servicio
@admin_bp.route('/servicio/nuevo', methods=['GET', 'POST'])
@admin_required
def nuevo_servicio():
    if request.method == 'POST':
        nombre = request.form.get('nombre')
        descripcion = request.form.get('descripcion')
        duracion = request.form.get('duracion')
        precio = request.form.get('precio')
        nuevo_servicio = Servicio(nombre=nombre, descripcion=descripcion, duracion=duracion, precio=precio)
        db.session.add(nuevo_servicio)
        db.session.commit()
        flash('Nuevo servicio creado con éxito', 'success')
        return redirect(url_for('admin.servicios'))
    return render_template('admin/nuevo_servicio.html')

# Crear Nueva Reserva
@admin_bp.route('/reserva/nueva', methods=['GET', 'POST'])
@admin_required
def nueva_reserva():
    if request.method == 'POST':
        usuario_id = request.form.get('usuario_id')
        vehiculo_id = request.form.get('vehiculo_id')
        servicio_id = request.form.get('servicio_id')
        slot_id = request.form.get('slot_id')
        problema = request.form.get('problema')
        fecha_hora = request.form.get('fecha_hora')
        nueva_reserva = Reserva(usuario_id=usuario_id, vehiculo_id=vehiculo_id, servicio_id=servicio_id, slot_id=slot_id, problema=problema, fecha_hora=fecha_hora)
        db.session.add(nueva_reserva)
        db.session.commit()
        flash('Nueva reserva creada con éxito', 'success')
        return redirect(url_for('admin.reservas'))
    return render_template('admin/nueva_reserva.html')

# Editar Servicio
@admin_bp.route('/servicio/editar/<int:servicio_id>', methods=['GET', 'POST'])
@admin_required
def editar_servicio(servicio_id):
    servicio = Servicio.query.get_or_404(servicio_id)
    if request.method == 'POST':
        servicio.nombre = request.form['nombre']
        servicio.descripcion = request.form['descripcion']
        servicio.duracion = request.form['duracion']
        servicio.precio = request.form['precio']
        db.session.commit()
        flash('Servicio actualizado correctamente.', 'success')
        return redirect(url_for('admin.servicios'))
    return render_template('admin/editar_servicio.html', servicio=servicio)

# Editar Cliente
@admin_bp.route('/cliente/editar/<int:cliente_id>', methods=['GET', 'POST'])
@admin_required
def editar_cliente(cliente_id):
    cliente = Usuario.query.get_or_404(cliente_id)
    if request.method == 'POST':
        cliente.nombre = request.form['nombre']
        cliente.apellido = request.form['apellido']
        cliente.email = request.form['email']
        cliente.telefono = request.form['telefono']
        cliente.direccion = request.form['direccion']
        cliente.genero = request.form['genero']
        cliente.fecha_nacimiento = request.form['fecha_nacimiento']
        cliente.pais = request.form['pais']
        
        # Actualizar la información de los vehículos
        for vehiculo_id, vehiculo_data in request.form.items():
            if vehiculo_id.startswith('vehiculo'):
                vid = int(vehiculo_id.split('[')[1].split(']')[0])
                vehiculo = Vehiculo.query.get(vid)
                vehiculo.marca = request.form[f"vehiculo[{vid}][marca]"]
                vehiculo.modelo = request.form[f"vehiculo[{vid}][modelo]"]
                vehiculo.año = request.form[f"vehiculo[{vid}][anio]"]

        db.session.commit()
        flash('Cliente y vehículos actualizados correctamente.', 'success')
        return redirect(url_for('admin.clientes'))
    
    return render_template('admin/editar_cliente.html', cliente=cliente)

# Editar Reserva
@admin_bp.route('/reserva/editar/<int:reserva_id>', methods=['GET', 'POST'])
@admin_required
def editar_reserva(reserva_id):
    reserva = Reserva.query.get_or_404(reserva_id)
    if request.method == 'POST':
        reserva.estado = request.form['estado']
        db.session.commit()
        flash('Reserva actualizada correctamente.', 'success')
        return redirect(url_for('admin.reservas'))
    return render_template('admin/editar_reserva.html', reserva=reserva)

# Eliminar Servicio
@admin_bp.route('/servicio/eliminar/<int:servicio_id>', methods=['POST'])
@admin_required
def eliminar_servicio(servicio_id):
    servicio = Servicio.query.get_or_404(servicio_id)
    db.session.delete(servicio)
    db.session.commit()
    flash('Servicio eliminado correctamente.', 'success')
    return redirect(url_for('admin.servicios'))

# Eliminar Cliente
@admin_bp.route('/cliente/eliminar/<int:cliente_id>', methods=['POST'])
@admin_required
def eliminar_cliente(cliente_id):
    cliente = Usuario.query.get_or_404(cliente_id)
    try:
        # Eliminar reservas asociadas
        reservas = Reserva.query.filter_by(usuario_id=cliente_id).all()
        for reserva in reservas:
            db.session.delete(reserva)

        # Eliminar vehículos asociados
        vehiculos = Vehiculo.query.filter_by(usuario_id=cliente_id).all()
        for vehiculo in vehiculos:
            db.session.delete(vehiculo)

        db.session.delete(cliente)
        db.session.commit()
        flash('Cliente eliminado correctamente.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error al eliminar el cliente: {str(e)}', 'danger')
    return redirect(url_for('admin.clientes'))
