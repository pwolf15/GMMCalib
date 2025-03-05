from dash_extensions import WebSocket
from dash_extensions.enrich import html, dcc, Output, Input, State, DashProxy
import dash
import requests
import plotly.graph_objects as go

# ✅ Load static point cloud data via HTTP request
STATIC_FIG1 = go.Figure()
try:
    response = requests.get("http://127.0.0.1:5000/static_data")
    if response.status_code == 200:
        static_data = response.json()["plot1"]
        for i, cloud in enumerate(static_data):
            STATIC_FIG1.add_trace(go.Scatter3d(
                x=[p[0] for p in cloud],
                y=[p[1] for p in cloud],
                z=[p[2] for p in cloud],
                mode="markers",
                marker=dict(size=3),
                name=f"Point Cloud {i+1}"
            ))
        STATIC_FIG1.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
except Exception as e:
    print(f"Error loading static data: {e}")

# ✅ Client-side function to update only the **dynamic plot**
update_graph = """function(msg, prev_fig2) {
    console.log("WebSocket message received:", msg);  
    if (!msg) { 
        return {};  // ✅ Ensure a valid object is returned
    } 

    const data = JSON.parse(msg.data);
    let fig2 = prev_fig2;

    if (data.plot2) {
        const x2 = data.plot2.map(p => p[0]), y2 = data.plot2.map(p => p[1]), z2 = data.plot2.map(p => p[2]);
        fig2 = {
            data: [{
                x: x2, y: y2, z: z2, mode: "markers", type: "scatter3d",
                marker: { size: 5, color: z2, colorscale: "Plasma" }
            }],
            layout: { uirevision: 'constant', scene: {xaxis: {title: "X2"}, yaxis: {title: "Y2"}, zaxis: {title: "Z2"}} }
        };
    }

    return fig2;
}""";

# ✅ Dash app layout (two separate windows)
app = DashProxy(__name__)
app.layout = html.Div([
    # ✅ First Window (Static Point Clouds)
    html.Div([
        html.H2("Static Point Clouds (First Window)"),
        dcc.Graph(id="graph1", figure=STATIC_FIG1),
    ], style={"width": "50%", "display": "inline-block"}),

    # ✅ Second Window (Dynamic Updates)
    html.Div([
        html.H2("Dynamic Point Cloud Updates (Second Window)"),
        WebSocket(id="ws", url="ws://127.0.0.1:5000/dynamic_data"),
        dcc.Graph(id="graph2")  # Dynamic
    ], style={"width": "50%", "display": "inline-block"})
])

# ✅ Clientside callback to update **only the dynamic plot**
app.clientside_callback(update_graph, 
                        [Output("graph2", "figure")], 
                        [Input("ws", "message")], 
                        [State("graph2", "figure")])  

if __name__ == "__main__":
    app.run_server(debug=True)
