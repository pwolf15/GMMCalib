import os
import json
import asyncio
import numpy as np
import multiprocessing
from quart import Quart, websocket, jsonify, send_file, send_from_directory
from hypercorn.asyncio import serve
from hypercorn.config import Config

class WebSocketServer:
    def __init__(self, calibration_instance, port=5000):
        """
        WebSocket Server to serve static and dynamic point clouds.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))  
        self.templates_dir = os.path.join(base_dir, "..", "templates")  
        self.static_dir = os.path.join(base_dir, "..", "static")  

        self.app = Quart(__name__, static_folder=self.static_dir)  
        self.calibration = calibration_instance
        self.latest_dynamic_data = {"Xin": None, "X": None}  
        self.clients = set()  
        self.port = port
        self.loop = asyncio.new_event_loop()  # ✅ Ensure we have an event loop for background tasks
        self.config_data = self.calibration.read_config()

        # ✅ Register HTTP Routes
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/static_data", "static_data", self.static_data)
        self.app.add_url_rule("/initial_calib_data", "initial_calib_data", self.get_initial_calib_data)
        self.app.add_url_rule("/config_data", "config_data", self.get_config_data)

        # ✅ Serve Static Files
        @self.app.route('/static/<path:filename>')
        async def static_files(filename):
            return await send_from_directory(self.static_dir, filename)

        @self.app.websocket("/ws")
        async def ws():
            """Handles WebSocket connection and continuously sends dynamic updates."""
            print("🔌 WebSocket client connected.")
            self.clients.add(websocket)

            # ✅ Start two concurrent tasks:
            sender = asyncio.create_task(self.sending())
            receiver = asyncio.create_task(self.receiving())

            # ✅ Run both tasks simultaneously
            await asyncio.gather(sender, receiver)

            # ✅ Cleanup when connection closes
            self.clients.discard(websocket)
            print("🔌 WebSocket client disconnected.")

    async def sending(self):
        """Continuously sends dynamic point cloud updates to WebSocket clients."""
        try:
            print("sending started!!!!!!!!!!!!")
            while True:
                if self.latest_dynamic_data["X"] is not None:
                    message = json.dumps({"plot2": self.latest_dynamic_data})
                    print(f"📤 Sending WebSocket update: {message[:200]}")
                    await websocket.send(message)
                await asyncio.sleep(0.1)  # ✅ Avoids excessive CPU usage
        except Exception as e:
            print(f"❌ Error in WebSocket sender: {e}")

    async def receiving(self):
        """Handles incoming WebSocket messages."""
        try:
            while True:
                message = await websocket.receive()
                print(f"📩 Received WebSocket message: {message}")  # ✅ Debug incoming messages
        except Exception as e:
            print(f"❌ Error in WebSocket receiver: {e}")

    async def index(self):
        """Serve the static HTML page."""
        return await send_file(os.path.join(self.templates_dir, "index.html"))

    async def static_data(self):
        """Serve the initial static point cloud data."""
        return jsonify(self.calibration.get_static_data())

    async def get_initial_calib_data(self):
        """Serve the initial dynamic point cloud data with Xin and X."""
        return jsonify(self.calibration.get_initial_calib_data())

    async def get_config_data(self):
        """Serve the configuration data."""
        return jsonify(self.config_data)

    def update_dynamic_geometry(self, fig_index, trace_index, new_points):
        """
        Updates the dynamic geometry (X) for WebSocket broadcasting.
        """
        print(f"📌 update_dynamic_geometry() called with shape: {new_points.shape}")

        if not isinstance(new_points, np.ndarray):
            print("⚠️ Warning: new_points is not a NumPy array. Converting...")
            new_points = np.array(new_points)

        if new_points.ndim != 2 or new_points.shape[1] != 3:
            print(f"❌ Error: new_points must be (N,3). Got shape {new_points.shape}")
            return

        self.latest_dynamic_data["X"] = {
            "x": new_points[:, 0].tolist(),
            "y": new_points[:, 1].tolist(),
            "z": new_points[:, 2].tolist(),
        }

        print(f"✅ Updated dynamic geometry for Figure {fig_index}, Trace {trace_index}, Shape: {new_points.shape}")

    async def broadcast_update(self):
        """Sends the latest dynamic point cloud data to all WebSocket clients."""
        print("here!")
        if not self.clients:
            print("⚠️ No active WebSocket clients to send updates.")
            return

        message = json.dumps({"plot2": self.latest_dynamic_data})
        print(f"📤 Attempting to send WebSocket update to {len(self.clients)} clients.")

        disconnected_clients = set()

        for client in self.clients:
            try:
                await client.send(message)
                print("✅ Sent WebSocket update to client.")
            except Exception as e:
                print(f"❌ Failed to send WebSocket update: {e}")
                disconnected_clients.add(client)

        # ✅ Remove disconnected clients
        self.clients -= disconnected_clients

    def run(self):
        """Run Quart with Hypercorn in an event loop."""
        config = Config()
        config.bind = [f"0.0.0.0:{self.port}"]  
        print(f"🚀 WebSocket server running on port {self.port}")

        asyncio.set_event_loop(self.loop)  # ✅ Set the event loop before running
        self.loop.run_until_complete(serve(self.app, config))

    def start_server(self):
        """Run the WebSocket server in a separate process to prevent signal conflicts."""
        self.loop = asyncio.new_event_loop()  # ✅ Explicitly create an event loop
        asyncio.set_event_loop(self.loop)
        process = multiprocessing.Process(target=self.run, daemon=True)
        process.start()
        print("✅ WebSocket server started in separate process.")
