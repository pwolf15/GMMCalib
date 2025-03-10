import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the CSV file
data_type = 'sim_data'
csv_file_path = f"results/{data_type}_results.csv"
df = pd.read_csv(csv_file_path)

# Get unique values of sequenceLength (number of observations) from the dataset
unique_observations = sorted(df["sequenceLength"].unique())

# Ensure at least one row, otherwise exit early
if len(unique_observations) == 0:
    raise ValueError("No unique observation values found in the dataset. Check data filtering.")

# Define marker colors based on useNoise (0 = blue, 1 = red)
color_map = {0: "blue", 1: "red"}

# Define marker symbols based on fixCentroids (0 = circle, 1 = square)
symbol_map = {0: "circle", 1: "square"}

# Define x-axis labels for Euler angles and Translation errors
euler_labels = ["Δφ (Roll)", "Δθ (Pitch)", "Δψ (Yaw)"]
translation_labels = ["Δx", "Δy", "Δz"]

# Create a single figure with multiple rows (one for each unique number of observations)
# Create a single figure with multiple rows (one for each unique observation count)
# Create a single figure with multiple rows (one for each unique observation count)
subplot_titles = []
for obs in unique_observations:
    subplot_titles.append(f"Euler Angle Errors (Sequence Length = {obs})")
    subplot_titles.append(f"Translation Errors (Sequence Length = {obs})")

fig_combined = make_subplots(
    rows=len(unique_observations), cols=2,
    subplot_titles=subplot_titles,
    vertical_spacing=0.1  # Adjust spacing between rows
)


# Define unique legend labels
legend_labels = {
    (0, 0): "Noise Off, FixCentroids Off",
    (0, 1): "Noise Off, FixCentroids On",
    (1, 0): "Noise On, FixCentroids Off",
    (1, 1): "Noise On, FixCentroids On"
}

# Track legend items to ensure only 4 legend entries
legend_added = set()

# Iterate over each unique value of num observations (one per row)
for row_idx, obs in enumerate(unique_observations, start=1):
    df_obs = df[df["sequenceLength"] == obs].copy()

    # Iterate over the four unique combinations of useNoise and fixCentroids
    for noise, centroid in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        df_subset = df_obs[(df_obs["useNoise"] == noise) & (df_obs["fixCentroids"] == centroid)]

        # Add data points for Euler Angle Errors (left column)
        for i, angle in enumerate(["eulerRollRad", "eulerPitchRad", "eulerYawRad"]):
            fig_combined.add_trace(go.Scatter(
                x=[euler_labels[i]] * len(df_subset),
                y=df_subset[angle],
                mode='markers',
                marker=dict(color=color_map[noise], symbol=symbol_map[centroid], size=12),
                hovertext=df_subset.apply(lambda row: f"Iter: {row['numIterations']}, "
                                                      f"Sequence Length: {row['sequenceLength']}, "
                                                      f"Noise: {row['useNoise']}, "
                                                      f"Fix: {row['fixCentroids']}", axis=1),
                name=legend_labels[(noise, centroid)] if (noise, centroid) not in legend_added else None,
                legendgroup="shared_legend",
                showlegend=(noise, centroid) not in legend_added
            ), row=row_idx, col=1)
            legend_added.add((noise, centroid))

    # Reset legend tracking for translation plot, but use the same legend entries
    for noise, centroid in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        df_subset = df_obs[(df_obs["useNoise"] == noise) & (df_obs["fixCentroids"] == centroid)]

        # Add data points for Translation Errors (right column)
        for i, trans in enumerate(["deltaTx", "deltaTy", "deltaTz"]):
            fig_combined.add_trace(go.Scatter(
                x=[translation_labels[i]] * len(df_subset),
                y=df_subset[trans],
                mode='markers',
                marker=dict(color=color_map[noise], symbol=symbol_map[centroid], size=12),
                hovertext=df_subset.apply(lambda row: f"Iter: {row['numIterations']}, "
                                                      f"Sequence Length: {row['sequenceLength']}, "
                                                      f"Noise: {row['useNoise']}, "
                                                      f"Fix: {row['fixCentroids']}", axis=1),
                name=None,  # Do not add duplicate names in legend
                legendgroup="shared_legend",
                showlegend=False  # No additional legend entries for the second plot
            ), row=row_idx, col=2)

# Update layout
fig_combined.update_layout(
    title=f"Impact of noise, fixing GMM centroids across sequence lengths on GMMCalib for {data_type}",
    height=500 * len(unique_observations),  # Dynamically scale based on number of rows
    width=1400,
    showlegend=True
)

# Save the combined plot as a single image
fig_combined.write_image(f"results/{data_type}_results.png")

print(f"All plots saved in a single image: 'results/{data_type}_results.png'.")
