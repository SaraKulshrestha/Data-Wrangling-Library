# My Data Wrangling Library for Machine Learning Projects

Welcome to my Data Wrangling Library, a tool I’ve designed to simplify and accelerate the data preprocessing journey for machine learning projects. Whether you're cleaning messy datasets, engineering new features, or handling imbalanced data, this library brings all the essential tools into one modular and reusable package. The idea is to help streamline the data preparation phase so you can focus more on building models and extracting insights.

# Key Features

1. Data Profiling
Quickly get to know your dataset with detailed information, missing values overview, skewness, and basic statistics.
2. Data Cleaning
Remove duplicates, handle missing values with custom strategies like mean/median, and detect outliers using both Z-Score and IQR (Interquartile Range) methods.
3. Feature Engineering
Encoding: Support for One-Hot Encoding, Label Encoding, and Ordinal Encoding.
Scaling: Normalize or standardize your features using StandardScaler or MinMaxScaler.
Polynomial Feature Creation: Add polynomial features to enhance model complexity.
Box-Cox Transformation: Normalize skewed data for better model performance.
Feature Selection: Easily select top features using statistical techniques like Chi-Square.
4. Data Splitting & Resampling
Easily split data into train and test sets, with options for stratification and handling imbalanced datasets using SMOTE.
5. Visualization Tools
Visualize distributions, explore feature relationships with pair plots, and check correlations via a heatmap.
6. Cross-Validation
Perform K-Fold cross-validation splits to ensure robust evaluation of your model.
Why I Built This
I designed this library during my machine learning journey when I realized that managing data transformation tasks across different projects can become repetitive and time-consuming. I wanted a personalized, flexible tool to automate much of the data wrangling process while keeping it clean and understandable for anyone who picks up my code later.


# How to Use
import pandas as pd
from datawrangler import DataWrangler

**Load your dataset**
df = pd.read_csv("your_dataset.csv")

** Initialize my Data Wrangler**
data_wrangler = DataWrangler(df)

** 1. Profile the dataset**
data_wrangler.data_profile()

** 2. Clean the data**
data_wrangler.clean_data()
data_wrangler.fill_missing(strategy="mean")

** 3. Encode categorical features**
data_wrangler.encode_categorical(columns=["category_column"])

** 4. Scale numerical features**
data_wrangler.scale_features(columns=["numerical_column"], scaling_type="standard")

** 5. Add polynomial features**
data_wrangler.add_polynomial_features(columns=["numerical_column"], degree=2)

** 6. Apply Box-Cox transformation**
data_wrangler.box_cox_transform(columns=["skewed_column"])

** 7. Split the data**
X_train, X_test, y_train, y_test = data_wrangler.split_data(target_column="target", stratify=True, handle_imbalance=True)

** 8. Visualize data distributions and correlations**
data_wrangler.plot_distributions(columns=["numerical_column"])
data_wrangler.correlation_matrix()

# Key Methods
data_profile(): Get a detailed summary of your dataset.
clean_data(): Remove duplicates and clean missing values.
fill_missing(strategy, columns): Impute missing values using different strategies.
remove_outliers(columns, z_thresh): Detect and remove outliers based on Z-Score.
remove_outliers_iqr(column): Remove outliers using the IQR method.
encode_categorical(columns): One-Hot Encode categorical columns.
label_encode(column): Label Encode categorical data.
ordinal_encode(column, categories): Ordinally encode categories based on order.
scale_features(columns, scaling_type): Scale features using either standard or min-max scaling.
add_polynomial_features(columns, degree): Generate polynomial features for model enrichment.
box_cox_transform(columns): Apply Box-Cox transformation to normalize skewed data.
split_data(target_column, test_size, stratify, handle_imbalance): Split your data for model training and testing.
plot_distributions(columns): Visualize data distributions.
correlation_matrix(): Create a heatmap of feature correlations.
k_fold_split(X, y, n_splits): Generate K-Fold splits for cross-validation.

# My Journey Behind the Library
The motivation for building this tool came from various data science projects where I had to repeatedly perform tedious data wrangling tasks. By consolidating these into a single framework, I now have a flexible, reusable solution that works for a wide range of machine learning projects—whether it's financial analysis or anomaly detection.



# Dependencies
pandas: For data manipulation.
numpy: For numerical computations.
scikit-learn: For preprocessing, feature selection, and evaluation.
seaborn and matplotlib: For data visualization.
imbalanced-learn: For balancing imbalanced datasets (SMOTE).
scipy: For statistical operations and outlier handling.

