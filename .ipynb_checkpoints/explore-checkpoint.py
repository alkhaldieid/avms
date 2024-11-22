import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
sales_data = pd.read_csv('data/snp_dld_2024_transactions.csv')
rentals_data = pd.read_csv('data/snp_dld_2024_rents.csv')

# Display basic info
print("Sales Data Info:")
print(sales_data.info())
print("\nRentals Data Info:")
print(rentals_data.info())

# Check for missing values
print("\nMissing Values in Sales Data:")
print(sales_data.isnull().sum())
print("\nMissing Values in Rentals Data:")
print(rentals_data.isnull().sum())

# Basic statistics
print("\nSales Data Statistics:")
print(sales_data.describe(include='all'))
print("\nRentals Data Statistics:")
print(rentals_data.describe(include='all'))

# Visualize distributions of numerical features
numerical_features = sales_data.select_dtypes(include=['float64', 'int64']).columns
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(sales_data[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature} (Sales Data)")
    plt.show()

# Visualize correlations in sales data
correlation_matrix = sales_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix (Sales Data)")
plt.show()

# Check for outliers using boxplots
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=sales_data[feature])
    plt.title(f"Boxplot of {feature} (Sales Data)")
    plt.show()

# Categorical feature analysis
categorical_features = sales_data.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sales_data[feature].value_counts().plot(kind='bar')
    plt.title(f"Value Counts of {feature} (Sales Data)")
    plt.xticks(rotation=45)
    plt.show()

# Compare sales and rental prices
if 'price' in sales_data.columns and 'price' in rentals_data.columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(sales_data['price'], label='Sales Price', shade=True)
    sns.kdeplot(rentals_data['price'], label='Rental Price', shade=True)
    plt.title("Sales vs Rentals Price Distribution")
    plt.legend()
    plt.show()

# Save cleaned data for further analysis
sales_data.to_csv('data/cleaned_sales_data.csv', index=False)
rentals_data.to_csv('data/cleaned_rentals_data.csv', index=False)

print("EDA complete. Cleaned datasets saved.")
