import dash
from dash import dcc, html, Output, Input, State
import plotly.graph_objects as go
import numpy as np
import threading
import time

class Live3DVisualizerPlotly:
    def __init__(self, figures_config):
        """
        Initializes a live 3D visualizer using Dash & Plotly with multiple configurable figures.
        
        Parameters:
        - `figures_config`: A list of dictionaries, each defining a figure with:
            - `"title"`: Initial title of the figure (can be updated).
            - `"traces"`: List of trace dictionaries, each containing:
                - `"points"`: Initial NumPy array (N, 3)
                - `"color"`: RGB color string (e.g., "red", "blue")
                - `"size"`: Marker size
                - `"name"`: Trace name
                - `"is_static"`: True for static traces, False for dynamic traces
        """
        self.figures_config = figures_config
        self.camera_views = [None] * len(figures_config)  # Store camera settings for each figure
        self.titles = [fig["title"] for fig in figures_config]  # Store dynamic titles

        # Initialize Dash app
        import os
        assets_path = os.path.join(os.path.dirname(__file__), "../assets")  # Ensure correct path
        self.app = dash.Dash(__name__, assets_folder=assets_path)

        # Store traces as a mutable structure (to allow updates)
        self.dynamic_traces = [
            [trace["points"] for trace in fig["traces"] if not trace["is_static"]]
            for fig in figures_config
        ]

        self.app.layout = html.Div([
            dcc.Store(id="figures-store", data=self.figures_config),
            dcc.Store(id="camera-store", data={}),
            dcc.Store(id="legend-store", data={}),
            
            html.Div([
                html.Div([
                    html.H4(id=f"figure-title-{i}", children=fig["title"], style={"text-align": "center"}),
                    dcc.Graph(id=f"live-3d-plot-{i}", style={"width": "100%", "height": "400px"})
                ], className="grid-item") for i, fig in enumerate(self.figures_config)
            ], className="grid-container"),

            dcc.Interval(id="interval-update", interval=2000, n_intervals=0)
        ])

        # Create callbacks for each figure
        # for i in range(len(figures_config)):
        #     self.app.callback(
        #         dash.Output(f"live-3d-plot-{i}", "figure"),
        #         dash.Output(f"figure-title-{i}", "children"),
        #         [dash.Input("interval-update", "n_intervals")]
        #     )(self._update_plot(i))
        # Register a single batch callback for all figures
        self.app.callback(
          [Output(f"live-3d-plot-{i}", "figure") for i in range(len(self.figures_config))] +
          [Output(f"figure-title-{i}", "children") for i in range(len(self.figures_config))] +
          [Output("figures-store", "data"), Output("camera-store", "data"), Output("legend-store", "data")],
          Input("interval-update", "n_intervals"),
          State("figures-store", "data"),
          State("camera-store", "data"),
          State("legend-store", "data")
        )(self.batch_update)

        # Run Dash in a separate thread
        self.thread = threading.Thread(target=self.app.run_server, kwargs={'debug': False, 'use_reloader': False})
        self.thread.start()

    def batch_update(self, n, stored_figures_config, stored_camera, stored_legend):
        """Efficiently updates all figures while preserving camera view & trace state."""
        updated_figures = []
        updated_titles = []
        new_camera = {}
        new_legend = {}

        for i, fig in enumerate(self.figures_config):
            figure = go.Figure()

            for trace in fig["traces"]:
                points = trace["points"]
                if not trace["is_static"]:
                    points = self.dynamic_traces[i].pop(0)  # Get updated points
                    self.dynamic_traces[i].append(points)  # Cycle back

                figure.add_trace(go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode="markers", marker=dict(color=trace["color"], size=trace["size"]),
                    name=trace["name"]
                ))

            # 🔹 Preserve Camera Settings
            if str(i) in stored_camera:
                figure.update_layout(scene_camera=stored_camera[str(i)])

            # 🔹 Preserve Trace Visibility
            if str(i) in stored_legend:
                for j, trace in enumerate(figure.data):
                    trace.visible = stored_legend[str(i)].get(str(j), True)

            # 🔹 Store camera & legend state for next update
            new_camera[str(i)] = figure.layout.scene.camera
            new_legend[str(i)] = {str(j): trace.visible for j, trace in enumerate(figure.data)}

            updated_figures.append(figure)
            updated_titles.append(self.titles[i])

        return updated_figures + updated_titles + [stored_figures_config, new_camera, new_legend]
      
    # def _update_plot(self, fig_index):
    #     """ Returns a callback function for updating a specific figure and its title. """
    #     def update(n):
    #         fig = go.Figure()

    #         for trace in self.figures_config[fig_index]["traces"]:
    #             points = trace["points"]
                
    #             # Update only dynamic traces
    #             if not trace["is_static"]:
    #                 points = self.dynamic_traces[fig_index].pop(0)  # Get latest dynamic points
    #                 self.dynamic_traces[fig_index].append(points)  # Recycle for next update

    #             fig.add_trace(go.Scatter3d(
    #                 x=points[:, 0], y=points[:, 1], z=points[:, 2],
    #                 mode="markers", marker=dict(color=trace["color"], size=trace["size"]),
    #                 name=trace["name"]
    #             ))

    #         # Preserve camera settings
    #         if self.camera_views[fig_index]:
    #             fig.update_layout(scene_camera=self.camera_views[fig_index])

    #         # Default layout settings
    #         fig.update_layout(
    #             scene=dict(xaxis=dict(visible=True), yaxis=dict(visible=True), zaxis=dict(visible=True)),
    #             margin=dict(l=0, r=0, t=0, b=0),
    #             width=600, height=400,
    #         )

    #         return fig, self.titles[fig_index]  # Return updated figure and title

    #     return update  # Return the callback function

    def update_dynamic_geometry(self, fig_index, trace_index, new_points):
        """
        Updates the dynamic geometry for a specific figure and trace.

        Parameters:
        - `fig_index`: Index of the figure (0-based).
        - `trace_index`: Index of the trace within the figure (0-based).
        - `new_points`: New (N,3) NumPy array of updated points.
        """
        self.dynamic_traces[fig_index][trace_index] = new_points

    def update_title(self, fig_index, new_title):
        """
        Updates the title of a specific figure.

        Parameters:
        - `fig_index`: Index of the figure (0-based).
        - `new_title`: The new title to set.
        """
        self.titles[fig_index] = new_title  # Update the title
        print(f"Updated title for Figure {fig_index}: {new_title}")

    def close(self):
        """Stops the visualization (not implemented due to Dash limitations)."""
        print("Dash server is running; close browser manually.")

# ✅ Example Usage
if __name__ == "__main__":
    # Define figures with configurable traces and titles
    figures_config = [
        {
            "title": "First 3D Figure",
            "traces": [
                {"points": np.random.rand(100, 3), "color": "red", "size": 3, "name": "Static Trace", "is_static": True},
                {"points": np.random.rand(50, 3), "color": "blue", "size": 3, "name": "Dynamic Trace", "is_static": False}
            ]
        },
        {
            "title": "Second 3D Figure",
            "traces": [
                {"points": np.random.rand(100, 3), "color": "green", "size": 3, "name": "Static Trace 2", "is_static": True},
                {"points": np.random.rand(50, 3), "color": "purple", "size": 3, "name": "Dynamic Trace 2", "is_static": False}
            ]
        }
    ]

    # Initialize visualizer with multiple figures and titles
    visualizer = Live3DVisualizerPlotly(figures_config)

    # Simulate dynamic updates
    for i in range(50):
        new_dynamic_points_1 = np.random.rand(50, 3)
        new_dynamic_points_2 = np.random.rand(50, 3)

        visualizer.update_dynamic_geometry(fig_index=0, trace_index=0, new_points=new_dynamic_points_1)
        visualizer.update_dynamic_geometry(fig_index=1, trace_index=0, new_points=new_dynamic_points_2)

        # Change the title every 10 iterations
        if i % 10 == 0:
            visualizer.update_title(0, f"Updated Figure 1 - Step {i}")
            visualizer.update_title(1, f"Updated Figure 2 - Step {i}")

        time.sleep(1)  # Simulate time delay
