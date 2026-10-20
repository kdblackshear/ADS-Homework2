import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Load dataset
data = pd.read_csv('bike_sharing_hour.csv')

# Display the first 5 rows of the dataset
print(f"First 5 rows of the dataset:\n{data.head()}\n")
# Display last 5 rows of the dataset
print(f"Last 5 rows of the dataset:\n{data.tail()}\n")
# Display number of rows and columns in the dataset
print(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}\n")
# Display all column names in the dataset
print(f"Column names in the dataset:\n{data.columns.tolist()}\n")
# Display data types of each column
print(f"Data types of each column:\n{data.dtypes}\n")
# Display information about the dataset
print(f"Information about the dataset:\n{data.info()}\n")
# Display descriptive statistics of the dataset
print(f"Descriptive statistics of the dataset:\n{data.describe()}\n")
# Display number of missing values in each column
print(f"Number of missing values in each column:\n{data.isnull().sum()}\n")
# Display number of exact duplicate rows in the dataset
print(f"Number of exact duplicate rows in the dataset:\n{data.duplicated().sum()}\n")


# Create output folder if it doesn't exist
from pathlib import Path 
output_folder = Path('output')
output_folder.mkdir(parents=True, exist_ok=True)

# Create raw copy of the original dataframe
raw_df = data.copy()
# Create a clean copy of the original dataframe
clean_df = data.copy()

# Rename columns for better readability
clean_df.rename(columns={
    'dteday': 'date',
    'yr': 'year_code',
    'mnth': 'month',
    'hr': 'hour',
    'weathersit': 'weather_code',
    'temp': 'temperature_normalized',
    'atemp': 'apparent_temperature_normalized',
    'hum': 'humidity_normalized',
    'windspeed': 'wind_speed_normalized',
    'casual': 'casual_rentals',
    'registered': 'registered_rentals',
    'cnt': 'total_rentals',
}, inplace=True)

# Convert date to pandas datetime format
clean_df['date'] = pd.to_datetime(clean_df['date'], errors='coerce')
# Create year column containing 2011 or 2012
clean_df['year'] = clean_df['year_code'].apply(lambda x: 2011 if x == 0 else 2012)
# Create season_label column containing Spring, Summer, Fall, or Winter
season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
clean_df['season_label'] = clean_df['season'].map(season_mapping)
# Create weather_label column containing Clear/Partly Cloudy, Mist/Cloudy, Light Rain/Snow, or Heavy Rain/Snow/Fog
weather_mapping = {
    1: 'Clear/Partly Cloudy',
    2: 'Mist/Cloudy',
    3: 'Light Rain/Snow',
    4: 'Heavy Rain/Snow/Fog'
}
clean_df['weather_label'] = clean_df['weather_code'].map(weather_mapping)
# Create day_type column working day and non-working day
clean_df['day_type'] = clean_df['workingday'].apply(lambda x: 'Working Day' if x == 1 else 'Non-Working Day' if x == 0 else 'Unknown')
# Create temperature_c, apparent_temperature_c, humidity_percent, and wind_speed columns
clean_df['temperature_c'] = 47 * clean_df['temperature_normalized'] - 8
clean_df['apparent_temperature_c'] = 66 * clean_df['apparent_temperature_normalized'] - 16
clean_df['humidity_percent'] = 100 * clean_df['humidity_normalized']
clean_df['wind_speed'] = 67 * clean_df['wind_speed_normalized']

# Display the column names of the clean dataframe
print(f"Column names of the clean dataframe:\n{clean_df.columns.tolist()}\n")
# Display the data types of the clean dataframe
print(f"Data types of the clean dataframe:\n{clean_df.dtypes}\n")
# Display the first 5 rows of the clean dataframe
print(f"First 5 rows of the clean dataframe:\n{clean_df.head()}\n")
# Display the last 5 rows of the clean dataframe
print(f"Last 5 rows of the clean dataframe:\n{clean_df.tail()}\n")


# Verify that instant contains unique values
assert clean_df['instant'].is_unique, "The 'instant' column does not contain unique values."
print("The 'instant' column contains unique values.\n")
# Verify that hour is between 0 and 23
assert clean_df['hour'].between(0, 23).all(), "The 'hour' column contains values outside the range of 0 to 23."
print("The 'hour' column contains values between 0 and 23.\n")
# Verify that month is between 1 and 12
assert clean_df['month'].between(1, 12).all(), "The 'month' column contains values outside the range of 1 to 12."
print("The 'month' column contains values between 1 and 12.\n")
# Verify that season and weather_code contain only their documented codes
valid_season_codes = {1, 2, 3, 4}
if clean_df['season'].isin(valid_season_codes).all():
    print("The 'season' column contains only valid codes (1, 2, 3, 4).\n")
else:
    print("The 'season' column contains invalid codes.\n")
valid_weather_codes = {1, 2, 3, 4}
if clean_df['weather_code'].isin(valid_weather_codes).all():
    print("The 'weather_code' column contains only valid codes (1, 2, 3, 4).\n")
else:
    print("The 'weather_code' column contains invalid codes.\n")   
# Verify that holiday and workingday contain only 0 or 1
valid_binary_codes = {0, 1}
if clean_df['holiday'].isin(valid_binary_codes).all():
    print("The 'holiday' column contains only valid binary codes (0, 1).\n")
else:
    print("The 'holiday' column contains invalid codes.\n")
if clean_df['workingday'].isin(valid_binary_codes).all():
    print("The 'workingday' column contains only valid binary codes (0, 1).\n")
else:
    print("The 'workingday' column contains invalid codes.\n")
# Verify that all rental count counts are non-negative
if (clean_df['casual_rentals'] >= 0).all() and (clean_df['registered_rentals'] >= 0).all() and (clean_df['total_rentals'] >= 0).all():
    print("All rental count columns contain non-negative values.\n")
else:
    print("One or more rental count columns contain negative values.\n")
# Verify that tottal_rentals = casual_rentals + registered_rentals
assert (clean_df['total_rentals'] == clean_df['casual_rentals'] + clean_df['registered_rentals']).all(), "The 'total_rentals' column does not equal the sum of 'casual_rentals' and 'registered_rentals'."
print("The 'total_rentals' column equals the sum of 'casual_rentals' and 'registered_rentals'.\n")
# Verify that date, hour, season_label, weather_label, temperature_c, humidity_percent, wind_speed, and total_rentals contain no missing values
columns_to_check = ['date', 'hour', 'season_label', 'weather_label', 'temperature_c', 'humidity_percent', 'wind_speed', 'total_rentals']
if clean_df[columns_to_check].isnull().sum().sum() == 0:
    print("The following columns contain no missing values: date, hour, season_label, weather_label, temperature_c, humidity_percent, wind_speed, total_rentals.\n")
else:
    print("One or more of the following columns contain missing values: date, hour, season_label, weather_label, temperature_c, humidity_percent, wind_speed, total_rentals.\n")
# Verify that the prepared dataset contains no exact duplicate rows
assert clean_df.duplicated().sum() == 0, "The prepared dataset contains exact duplicate rows."
print("The prepared dataset contains no exact duplicate rows.\n")


# Calculate and display:
# the mean, median, standard deviation, minimum, maximum, and quartiles of the total_rentals column
print("Descriptive statistics for 'total_rentals':")
print(clean_df['total_rentals'].describe())
# the number of observations for each season and weather category
print("\nNumber of observations for each season:")
print(clean_df['season_label'].value_counts())
print("\nNumber of observations for each weather category:")
print(clean_df['weather_label'].value_counts())
# the mean and median total rentals for working days and non-working days
print("\nMean and median total rentals for working days and non-working days:")
working_day_stats = clean_df.groupby('day_type')['total_rentals'].agg(['mean', 'median'])
print(working_day_stats)
# the mean total rentals for each hour of the day
print("\nMean total rentals for each hour of the day:")
hourly_stats = clean_df.groupby('hour')['total_rentals'].mean()
print(hourly_stats)
# the number and mean total rentals grouped by both season_label and workingday
print("\nNumber and mean total rentals grouped by both season_label and workingday:")
season_workingday_stats = clean_df.groupby(['season_label', 'day_type'])['total_rentals'].agg(['count', 'mean'])
print(season_workingday_stats)

# Create and save visualizations:
# a histogram of the total_rentals column
plt.figure(figsize=(10, 6))
sns.histplot(clean_df['total_rentals'], bins=30, kde=True)
plt.title('Histogram of Total Rentals')
plt.xlabel('Total Rentals')
plt.ylabel('Frequency')
plt.savefig(output_folder / 'histogram_total_rentals.png')
# a boxplot of total_rentals by season
plt.figure(figsize=(10, 6))
sns.boxplot(x='season_label', y='total_rentals', data=clean_df)
plt.title('Boxplot of Total Rentals by Season')
plt.xlabel('Season')
plt.ylabel('Total Rentals')
plt.savefig(output_folder / 'boxplot_total_rentals_by_season.png')
# a line plot of mean total rentals by hour of the day, with separate lines for working days and non-working days
plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='total_rentals', hue='day_type', data=clean_df.groupby(['hour', 'day_type'])['total_rentals'].mean().reset_index())
plt.title('Mean Total Rentals by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Mean Total Rentals')
plt.legend(title='Day Type')
plt.savefig(output_folder / 'lineplot_mean_total_rentals_by_hour.png')
# a scatter plot of temperature in celsius vs total rentals, with color representing season
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temperature_c', y='total_rentals', hue='season_label', data=clean_df)
plt.title('Scatter Plot of Temperature (C) vs Total Rentals')
plt.xlabel('Temperature (C)')
plt.ylabel('Total Rentals')
plt.legend(title='Season')
plt.savefig(output_folder / 'scatterplot_temperature_vs_total_rentals.png')


# Create a boolean column named high_demand (true if total_rentals is >= dataset's 75th percentile)
clean_df['high_demand'] = clean_df['total_rentals'] >= clean_df['total_rentals'].quantile(0.75)
# Calculate and report:
# the 75th percentile threshold
high_demand_threshold = clean_df['total_rentals'].quantile(0.75)
print(f"\n75th percentile threshold for high demand: {high_demand_threshold}\n")
# P(high demand)
prob_high_demand = clean_df['high_demand'].mean()
print(f"Probability of high demand: {prob_high_demand}\n")
# P(high demand | working day)
num_high_demand_working_day = clean_df[(clean_df['day_type'] == 'Working Day') & (clean_df['high_demand'])].shape[0]
den_high_demand_working_day = clean_df[clean_df['day_type'] == 'Working Day'].shape[0]
print(f"Numerator for P(high demand | working day): {num_high_demand_working_day}\n")
print(f"Denominator for P(high demand | working day): {den_high_demand_working_day}\n")
prob_high_demand_working_day = clean_df[clean_df['day_type'] == 'Working Day']['high_demand'].mean()
print(f"Probability of high demand given working day: {prob_high_demand_working_day}\n")
# P(high demand | non-working day)
num_high_demand_non_working_day = clean_df[(clean_df['day_type'] == 'Non-Working Day') & (clean_df['high_demand'])].shape[0]
den_high_demand_non_working_day = clean_df[clean_df['day_type'] == 'Non-Working Day'].shape[0]
print(f"Numerator for P(high demand | non-working day): {num_high_demand_non_working_day}\n")
print(f"Denominator for P(high demand | non-working day): {den_high_demand_non_working_day}\n")
prob_high_demand_non_working_day = clean_df[clean_df['day_type'] == 'Non-Working Day']['high_demand'].mean()
print(f"Probability of high demand given non-working day: {prob_high_demand_non_working_day}\n")
# P(high demand | clear or partly cloudy weather)
num_high_demand_clear_weather = clean_df[(clean_df['weather_label'] == 'Clear/Partly Cloudy') & (clean_df['high_demand'])].shape[0]
den_high_demand_clear_weather = clean_df[clean_df['weather_label'] == 'Clear/Partly Cloudy'].shape[0]
print(f"Numerator for P(high demand | clear or partly cloudy weather): {num_high_demand_clear_weather}\n")
print(f"Denominator for P(high demand | clear or partly cloudy weather): {den_high_demand_clear_weather}\n")
prob_high_demand_clear_weather = clean_df[clean_df['weather_label'] == 'Clear/Partly Cloudy']['high_demand'].mean()
print(f"Probability of high demand given clear or partly cloudy weather: {prob_high_demand_clear_weather}\n")

# Create a Pearson correlation matrix (round 2 decimals) using the following columns: temperature_c, apparent_temperature_c, humidity_percent, wind_speed, and total_rentals
correlation_matrix = clean_df[['temperature_c', 'apparent_temperature_c', 'humidity_percent', 'wind_speed', 'total_rentals']].corr().round(2)
print("Pearson correlation matrix:")
print(correlation_matrix)
# Create annotated heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.savefig(output_folder / 'correlation_matrix_heatmap.png')
# Identify weather variable with the strongest absolute Pearson correlation with total_rentals
correlation_with_total_rentals = correlation_matrix['total_rentals'].drop('total_rentals')
strongest_correlation_variable = correlation_with_total_rentals.abs().idxmax()
strongest_correlation_value = correlation_with_total_rentals[strongest_correlation_variable]
print(f"The weather variable with the strongest absolute Pearson correlation with total_rentals is '{strongest_correlation_variable}' with a correlation value of {strongest_correlation_value}.")
# Report the correlation coefficient and interpret its direction and strength
if strongest_correlation_value > 0:
    direction = "positive"
else:
    direction = "negative"
strength = "strong" if abs(strongest_correlation_value) >= 0.7 else "moderate" if abs(strongest_correlation_value) >= 0.3 else "weak"
print(f"The correlation coefficient is {strongest_correlation_value}, indicating a {strength} {direction} correlation between '{strongest_correlation_variable}' and 'total_rentals'.")



