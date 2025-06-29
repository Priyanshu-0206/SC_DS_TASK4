# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

# Read Dataset
data_path = (r"C:\Users\Priyanshu\Downloads\US_Accidents_March23.csv.zip")
df = pd.read_csv(data_path)

print("Dataset Loaded. First 5 Rows:")
print(df.head())

# Basic cleaning: Handle missing values
df = df[['Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'Weather_Condition',
         'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Temperature(F)',
         'Humidity(%)', 'Pressure(in)', 'Traffic_Signal', 'Sunrise_Sunset', 'Distance(mi)']]

df.dropna(subset=['Start_Lat', 'Start_Lng', 'Weather_Condition', 'Visibility(mi)', 
                  'Temperature(F)', 'Humidity(%)'], inplace=True)

# Convert Start_Time to datetime properly (handles the earlier error)
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

# Remove any rows where datetime conversion failed
df.dropna(subset=['Start_Time'], inplace=True)

# Extract Hour, Month, Weekday
df['Hour'] = df['Start_Time'].dt.hour
df['Month'] = df['Start_Time'].dt.month
df['Weekday'] = df['Start_Time'].dt.dayofweek  # Monday=0, Sunday=6

# =====================================
# 1. Pattern: Severity by Weather
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Severity', hue='Weather_Condition', order=sorted(df['Severity'].unique()), dodge=False, legend=False)
plt.title('Severity by Weather Condition')
plt.xticks([0,1,2,3], ['Low', 'Moderate', 'High', 'Very High'])
plt.show()

# Top 10 Weather Conditions
top_weather = df['Weather_Condition'].value_counts().nlargest(10).index
plt.figure(figsize=(12,6))
sns.countplot(data=df[df['Weather_Condition'].isin(top_weather)], y='Weather_Condition', order=top_weather)
plt.title('Top 10 Weather Conditions during Accidents')
plt.show()

# =====================================
# 2. Pattern: Time of Day (Hour)
plt.figure(figsize=(12,6))
sns.countplot(x='Hour', data=df, hue='Hour', palette='viridis', dodge=False, legend=False)
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour (0-23)')
plt.ylabel('Number of Accidents')
plt.show()

# =====================================
# 3. Pattern: Weekday
plt.figure(figsize=(10,5))
sns.countplot(x='Weekday', data=df, hue='Weekday', palette='Set2', dodge=False, legend=False)
plt.title('Accidents by Weekday')
plt.xlabel('Weekday (0=Monday)')
plt.ylabel('Number of Accidents')
plt.show()

# =====================================
# 4. Pattern: Road Conditions (Distance)
plt.figure(figsize=(10,5))
sns.histplot(df['Distance(mi)'], bins=50, kde=True)
plt.title('Distribution of Accident Distance')
plt.xlabel('Distance (miles)')
plt.show()

# =====================================
# 5. Pattern: Visibility Impact
plt.figure(figsize=(10,5))
sns.boxplot(x='Severity', y='Visibility(mi)', data=df)
plt.title('Visibility vs Severity')
plt.show()

# =====================================
# 6. Pattern: Temperature Impact
plt.figure(figsize=(10,5))
sns.boxplot(x='Severity', y='Temperature(F)', data=df)
plt.title('Temperature vs Severity')
plt.show()

# =====================================
# 7. Map: Accident Hotspots (HeatMap)
heat_df = df[['Start_Lat', 'Start_Lng']].dropna()
heat_data = heat_df.values.tolist()

base_map = folium.Map(location=[37.0902, -95.7129], zoom_start=5)  # USA center
HeatMap(heat_data[:10000]).add_to(base_map)  # Limited for performance

# Save to HTML
base_map.save("Accident_Hotspots.html")
print("\nHeatmap saved as 'Accident_Hotspots.html'.")

# =====================================
# 8. Sunrise vs Sunset Impact
plt.figure(figsize=(8,5))
sns.countplot(x='Sunrise_Sunset', data=df)
plt.title('Accidents: Day vs Night')
plt.show()

# =====================================
# 9. Correlation Heatmap for Numerical Features
plt.figure(figsize=(12,8))
numerical_cols = ['Severity', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                  'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Distance(mi)']
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Numerical Features)')
plt.show()
