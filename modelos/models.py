from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Usuario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    apellido = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    telefono = db.Column(db.String(15), nullable=False)
    direccion = db.Column(db.String(255))
    ciudad = db.Column(db.String(100))
    profesion = db.Column(db.String(100))
    pais = db.Column(db.String(100))
    fecha_registro = db.Column(db.DateTime, default=db.func.current_timestamp())
    fecha_nacimiento = db.Column(db.Date)
    genero = db.Column(db.Enum('M', 'F', 'Otro'))
    preferencias_servicio = db.Column(db.JSON)
    rol = db.Column(db.Enum('usuario', 'administrador'), default='usuario')
    activo = db.Column(db.Boolean, default=True)
    estado = db.Column(db.String(50), default='inicio')
    password_hash = db.Column(db.String(128))
    vehiculos = db.relationship('Vehiculo', backref='usuario', lazy=True)
    reservas = db.relationship('Reserva', backref='usuario', lazy=True)
    comentarios_servicio = db.relationship('ComentarioServicio', backref='usuario', lazy=True)
    registros_usuario = db.relationship('RegistroUsuario', backref='usuario', lazy=True)
    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<Usuario {self.nombre} {self.apellido}>'

class Vehiculo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)
    marca = db.Column(db.String(50), nullable=False)
    modelo = db.Column(db.String(50), nullable=False)
    año = db.Column(db.Integer, nullable=False)
    reservas = db.relationship('Reserva', backref='vehiculo', lazy=True)

    def __repr__(self):
        return f'<Vehiculo {self.marca} {self.modelo}>'

class Servicio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    descripcion = db.Column(db.Text)
    duracion = db.Column(db.String(50))
    precio = db.Column(db.Numeric(10, 2))
    slots = db.relationship('Slot', backref='servicio', lazy=True)
    reservas = db.relationship('Reserva', backref='servicio', lazy=True)
    comentarios_servicio = db.relationship('ComentarioServicio', backref='servicio', lazy=True)

    def __repr__(self):
        return f'<Servicio {self.nombre}>'

class Slot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    servicio_id = db.Column(db.Integer, db.ForeignKey('servicio.id'), nullable=False)
    fecha = db.Column(db.Date, nullable=False)
    hora_inicio = db.Column(db.Time, nullable=False)
    hora_fin = db.Column(db.Time, nullable=False)
    reservado = db.Column(db.Boolean, default=False)
    reservas = db.relationship('Reserva', backref='slot', lazy=True)

    def __repr__(self):
        return f'<Slot {self.fecha} {self.hora_inicio}-{self.hora_fin}>'

class Reserva(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)
    vehiculo_id = db.Column(db.Integer, db.ForeignKey('vehiculo.id'), nullable=False)
    servicio_id = db.Column(db.Integer, db.ForeignKey('servicio.id'), nullable=False)
    slot_id = db.Column(db.Integer, db.ForeignKey('slot.id'), nullable=False)
    problema = db.Column(db.Text, nullable=False)
    fecha_hora = db.Column(db.DateTime, nullable=False)
    estado = db.Column(db.String(20), nullable=False, default='no realizado')
    registros_servicio = db.relationship('RegistroServicio', backref='reserva', lazy=True)

    def __repr__(self):
        return f'<Reserva {self.id} {self.fecha_hora}>'

class ComentarioServicio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)
    servicio_id = db.Column(db.Integer, db.ForeignKey('servicio.id'), nullable=False)
    comentario = db.Column(db.Text, nullable=False)
    analisis_sentimiento = db.Column(db.String(50))
    fecha_hora = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<ComentarioServicio {self.id} {self.fecha_hora}>'

class Repuesto(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    descripcion = db.Column(db.Text)
    precio = db.Column(db.Numeric(10, 2))
    stock = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Repuesto {self.nombre}>'

class RegistroUsuario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)
    tiempo_inicio = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    tiempo_fin = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f'<RegistroUsuario {self.id} {self.tiempo_inicio}>'

class RegistroServicio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    reserva_id = db.Column(db.Integer, db.ForeignKey('reserva.id'), nullable=False)
    tiempo_inicio = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    tiempo_fin = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f'<RegistroServicio {self.id} {self.tiempo_inicio}>'

class Interaccion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=True)
    mensaje_usuario = db.Column(db.Text, nullable=False)
    respuesta_bot = db.Column(db.Text, nullable=False)
    es_exitosa = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)

    def __repr__(self):
        return f'<Interaccion {self.id} {self.timestamp}>'
