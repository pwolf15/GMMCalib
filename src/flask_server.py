import os
from flask import Flask, render_template
from flask_socketio import SocketIO

# ✅ Get the absolute path to the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Ensure Flask knows where to find templates & static files
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, "../templates"), 
            static_folder=os.path.join(BASE_DIR, "../static"))

socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100 * 1024 * 1024)  # ✅ Enable WebSockets

@socketio.on("initial_positions")
def handle_point_cloud(data):
    """Handles point cloud data from the Python application and broadcasts it."""
    # print(f"📩 Received Point Cloud: {len(data['points'])} points")
    print("received initial positions")
    
    # ✅ Broadcast to all connected clients
    socketio.emit("initial_positions", data)

@socketio.on("gmm_means")
def handle_gmm_means(data):
    print("received gmm means")
    socketio.emit("gmm_means", data)

@socketio.on("registrations")
def handle_gmm_means(data):
    print("received registrations")
    socketio.emit("registrations", data)

@app.route('/')
def index():
    return render_template('index.html')  # ✅ Serve the front-end page

if __name__ == '__main__':
    print(f"🚀 Running Flask from: {BASE_DIR}")  # ✅ Debug output to verify path
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
