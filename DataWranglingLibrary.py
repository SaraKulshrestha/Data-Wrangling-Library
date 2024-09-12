import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PolynomialFeatures, PowerTransformer, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class DataWrangler:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    # 1. Data Profiling
    def data_profile(self):
        print("Dataset Info:")
        print(self.data.info())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nSkewness:")
        print(self.data.skew())
        print("\nDescription:")
        print(self.data.describe())
        return self.data

    # 2. Data Cleaning
    def clean_data(self, drop_duplicates=True, drop_na_thresh=None):
        if drop_duplicates:
            self.data.drop_duplicates(inplace=True)
        if drop_na_thresh:
            self.data.dropna(thresh=drop_na_thresh, inplace=True)
        return self.data
    
    def fill_missing(self, strategy="mean", columns=None):
        imputer = SimpleImputer(strategy=strategy)
        if columns:
            self.data[columns] = imputer.fit_transform(self.data[columns])
        else:
            self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        return self.data

    def remove_outliers(self, columns, z_thresh=3):
        for col in columns:
            self.data = self.data[(np.abs(stats.zscore(self.data[col])) < z_thresh)]
        return self.data

    def remove_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        self.data = self.data[~((self.data[column] < (Q1 - 1.5 * IQR)) | (self.data[column] > (Q3 + 1.5 * IQR)))]
        return self.data

    # 3. Feature Engineering
    def encode_categorical(self, columns):
        encoder = OneHotEncoder(sparse=False, drop="first")
        encoded_df = pd.DataFrame(encoder.fit_transform(self.data[columns]), columns=encoder.get_feature_names_out())
        self.data = self.data.drop(columns, axis=1)
        self.data = pd.concat([self.data, encoded_df], axis=1)
        return self.data
    
    def label_encode(self, column):
        encoder = LabelEncoder()
        self.data[column] = encoder.fit_transform(self.data[column])
        return self.data

    def ordinal_encode(self, column, categories):
        encoder = OrdinalEncoder(categories=[categories])
        self.data[column] = encoder.fit_transform(self.data[[column]])
        return self.data

    def scale_features(self, columns, scaling_type="standard"):
        scaler = StandardScaler() if scaling_type == "standard" else MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data

    def add_polynomial_features(self, columns, degree=2):
        poly = PolynomialFeatures(degree)
        poly_features = poly.fit_transform(self.data[columns])
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(columns))
        self.data = pd.concat([self.data.drop(columns, axis=1), poly_df], axis=1)
        return self.data

    def box_cox_transform(self, columns):
        pt = PowerTransformer(method='box-cox', standardize=True)
        self.data[columns] = pt.fit_transform(self.data[columns])
        return self.data
    
    def feature_selection(self, X, y, k=10):
        selector = SelectKBest(chi2, k=k)
        X_new = selector.fit_transform(X, y)
        return pd.DataFrame(X_new, columns=X.columns[selector.get_support()])

    # 4. Data Splitting with Stratification and SMOTE
    def split_data(self, target_column, test_size=0.2, stratify=False, handle_imbalance=False):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        if handle_imbalance:
            smote = SMOTE()
            X, y = smote.fit_resample(X, y)
        
        if stratify:
            return train_test_split(X, y, test_size=test_size, stratify=y)
        else:
            return train_test_split(X, y, test_size=test_size)

    # 5. Visualization Tools
    def plot_distributions(self, columns):
        for col in columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

    def correlation_matrix(self):
        plt.figure(figsize=(12, 10))
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    def pair_plot(self, columns):
        sns.pairplot(self.data[columns])
        plt.show()

    # 6. Cross-Validation Split
    def k_fold_split(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            yield X_train, X_test, y_train, y_test

# main function
if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    
    # Initialize the data wrangler
    data_wrangler = DataWrangler(df)
    
    # Profile data
    data_wrangler.data_profile()
    
    # Clean and fill missing data
    data_wrangler.clean_data()
    data_wrangler.fill_missing(strategy="mean")
    
    # Encode categorical features
    data_wrangler.encode_categorical(columns=["category_col"])
    
    # Scale features
    data_wrangler.scale_features(columns=["num_feature"])
    
    # Polynomial feature creation
    data_wrangler.add_polynomial_features(columns=["num_feature"], degree=2)
    
    # Box-Cox transformation for normalizing skewed data
    data_wrangler.box_cox_transform(columns=["skewed_feature"])
    
    # Split data
    X_train, X_test, y_train, y_test = data_wrangler.split_data(target_column="target", stratify=True, handle_imbalance=True)
    
    # Visualize distributions and correlation
    data_wrangler.plot_distributions(columns=["num_feature"])
    data_wrangler.correlation_matrix()
