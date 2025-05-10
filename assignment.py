# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Suppress warnings and set visual style
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("Dataset loaded successfully.\n")
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
print("First five rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# No missing values to clean in Iris, but here's how you'd handle them:
# df = df.dropna() or df.fillna(method='ffill')

# Task 2: Basic Data Analysis
print("\nDescriptive Statistics:")
print(df.describe())

# Group by species and get mean of numerical columns
grouped_means = df.groupby("target").mean()
print("\nMean values grouped by species (target):")
print(grouped_means)

# Replace numerical target with actual species names for readability
df['species'] = df['target'].apply(lambda x: iris_data.target_names[x])

# Task 3: Data Visualization
plt.figure(figsize=(14, 10))

# Line chart - Simulate a trend (cumulative sum as example)
plt.subplot(2, 2, 1)
df_sorted = df.sort_values(by='sepal length (cm)')
plt.plot(df_sorted.index, df_sorted['sepal length (cm)'].cumsum(), label='Cumulative Sepal Length')
plt.title('Line Chart: Cumulative Sepal Length')
plt.xlabel('Index')
plt.ylabel('Cumulative Sepal Length (cm)')
plt.legend()

# Bar chart - Average petal length per species
plt.subplot(2, 2, 2)
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Bar Chart: Avg Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')

# Histogram - Distribution of sepal length
plt.subplot(2, 2, 3)
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Histogram: Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')

# Scatter plot - Sepal vs. Petal length
plt.subplot(2, 2, 4)
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Scatter Plot: Sepal vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

# Adjust layout and show plots
plt.tight_layout()
plt.show()

# Findings and Observations
print("\nObservations:")
print("- Iris-virginica tends to have larger sepal and petal lengths.")
print("- Sepal length appears to increase steadily (as shown in the line chart).")
print("- Histogram shows a roughly normal distribution for sepal length.")
print("- Scatter plot shows a clear distinction between species based on sepal and petal lengths.")
