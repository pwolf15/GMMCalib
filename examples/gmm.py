import pandas as pd
from pybaseball import statcast
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Fetching MLB Statcast Data from June 1st, 2023
df = statcast(start_dt='2023-06-01', end_dt='2023-06-30')

# Step 2: Check the columns and data to understand its structure
print("Columns in the dataset:", df.columns)
print("First few rows of data:")
print(df.head())

# Step 3: Data Preprocessing - Focus on release speed and pitch type
df_clean = df[['player_name', 'release_speed', 'pitch_type']]

# Drop rows with missing data (if any)
df_clean = df_clean.dropna()

# Step 4: Check unique pitch types in the data
print("Unique pitch types in the data:")
print(df_clean['pitch_type'].unique())

# Step 5: Define a more accurate mapping of pitch types to names
pitch_type_mapping = {
    'FF': 'Four-Seam Fastball',
    'SI': 'Sinker',
    'SL': 'Slider',
    'CU': 'Curveball',
    'CH': 'Changeup',
    'FC': 'Cutter',
    'FS': 'Splitter',
    'KN': 'Knuckleball',
    'FT': 'Two-Seam Fastball',
    'FO': 'Forkball',
    'EP': 'Eephus'
}

# Map pitch types to names
df_clean['Pitch_Type_Name'] = df_clean['pitch_type'].map(pitch_type_mapping)

# Check if mapping worked, ensure there are no NaNs in 'Pitch_Type_Name'
print("Checking for NaNs in 'Pitch_Type_Name' after mapping:")
print(df_clean['Pitch_Type_Name'].isna().sum())

# Step 6: Convert categorical pitch_type into numerical values for clustering
df_clean['pitch_type_code'] = df_clean['pitch_type'].astype('category').cat.codes

# Step 7: Extract features for clustering
features = df_clean[['release_speed', 'pitch_type_code']]

# Step 8: Standardize the features (important for clustering)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 9: Apply Gaussian Mixture Model (GMM) for clustering
gmm = GaussianMixture(n_components=3, random_state=42)  # 3 clusters, you can change this value
df_clean['Cluster'] = gmm.fit_predict(features_scaled)

# Step 10: Reset index and drop duplicates to avoid the duplicate labels issue
df_clean = df_clean.reset_index(drop=True)  # Reset index to ensure no duplicates
df_clean = df_clean.drop_duplicates(subset=['player_name', 'release_speed', 'pitch_type'])  # Remove duplicates based on key columns

# Step 11: Visualize the clusters with the actual pitch types on the y-axis
plt.figure(figsize=(10, 6))

# Set the y-axis as categorical (pitch types) using sns.scatterplot
sns.scatterplot(x='release_speed', y='Pitch_Type_Name', hue='Cluster', palette='viridis', data=df_clean)

plt.title('Clustering of MLB Pitches Based on Release Speed and Pitch Type (June 1st - June 30th, 2023)')
plt.xlabel('Release Speed')
plt.ylabel('Pitch Type')
plt.show()

# Step 12: Show the means (centroids) of each cluster
cluster_means = pd.DataFrame(gmm.means_, columns=['release_speed', 'pitch_type_code'])
print("Cluster Means (Centroids):")
print(cluster_means)

# Step 13: Fastest and Slowest pitch speeds for each pitch type, along with the thrower (player_name)
# Now apply groupby to find the fastest and slowest pitches for each pitch type
fastest_slowest_with_players = df_clean.groupby('Pitch_Type_Name').apply(
    lambda x: pd.Series({
        'Fastest_Speed': x.loc[x['release_speed'].idxmax()],
        'Slowest_Speed': x.loc[x['release_speed'].idxmin()]
    })
).reset_index(drop=True)  # Ensure unique index

# Step 14: Print out the player names along with the fastest and slowest pitches
print("\nFastest and Slowest Release Speeds for Each Pitch Type with Players:")
for index, row in fastest_slowest_with_players.iterrows():
    pitch_type = row['Pitch_Type_Name']
    fastest = row['Fastest_Speed']
    slowest = row['Slowest_Speed']
    
    fastest_player = fastest['player_name']
    fastest_speed = fastest['release_speed']
    
    slowest_player = slowest['player_name']
    slowest_speed = slowest['release_speed']
    
    print(f"\nPitch Type: {pitch_type}")
    print(f"  Fastest: {fastest_player} - {fastest_speed} mph")
    print(f"  Slowest: {slowest_player} - {slowest_speed} mph")

# Step 15: Check the final data with mapped pitch type names
print("\nFinal Data with Mapped Pitch Types:")
print(df_clean.head())
