import socketio
import time

class SocketIOClient:

    def __init__(self, url):

        self.sio = socketio.Client()
        sio = self.sio

        @sio.event
        def connect():
            print("Connected to server")
        
        @sio.event
        def disconnect():
            print("Disconnected from socket server")
        
        @sio.on('server_response')
        def handle_response(data):
            print(f"Received data {data}")

        sio.connect(url)
    
    def emit(self, message, data):
        print(f'{message} data')
        self.sio.emit(message, data)


def send_point_cloud():
    """Simulates sending point cloud data to the Flask WebSocket server."""


if __name__ == '__main__':
    # ✅ Connect to Flask Socket.IO server
    sio.connect('http://127.0.0.1:5000')

    # ✅ Start sending data
    send_point_cloud()
