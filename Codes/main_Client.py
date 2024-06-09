from Client import Client_Creation
from flask_socketio import SocketIO, emit
app=Client_Creation()

if __name__ == '__main__':
    app.run(debug = True,host = '0.0.0.0', port = 8002)
