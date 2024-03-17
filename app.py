import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro
import base64  # Added base64 for encoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Custom color values
custom_background_color = "#f0f0f0"
custom_primary_color = "#ff5733"  # Orange
custom_secondary_color = "#333333"  # Dark Gray

# Apply styling with custom colors
st.markdown(
    f"""
    <style>
        body {{ background-color: {custom_background_color}; font-family: 'Arial', sans-serif; }}
        h1, .stApp {{ color: {custom_primary_color}; text-align: center; }}
        h2, h3, .stMarkdown {{ color: {custom_secondary_color}; }}
        .stTextInput, .stFileUploader, .stButton, .stTextArea {{ border-color: {custom_primary_color}; }}
        .stTextInput, .stFileUploader, .stButton:hover {{ background-color: {custom_primary_color}; color: white; }}
    </style>
    """, unsafe_allow_html=True
)

def load_and_display_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(('.csv', '.xls', '.xlsx')):
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    st.success("Data Loaded Successfully")
    return df

def display_basic_info(df):
    st.subheader("Data Overview:")
    st.write(df.head())

    st.subheader("Data Statistics:")
    st.write(df.describe())

def exploratory_data_analysis(df):
    st.subheader("Exploratory Data Analysis (EDA):")

    st.write("### Missing Values:")
    st.write(df.isnull().sum())

    st.write("### Data Types:")
    st.write(df.dtypes)

    st.write("### Correlation Matrix:")
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5)
    st.pyplot(plt.gcf())

    st.write("### Columns Normality Check:")
    for column in numeric_df.columns:
        _, p_value = shapiro(df[column].dropna())
        st.write(f"{column}: p-value = {p_value:.4f}")
        st.write("Normal distribution check: ", "Not Normal" if p_value < 0.05 else "Normal")

def data_analysis_charts(df):
    st.subheader("Data Analysis Charts and Graphs:")

    selected_columns = st.multiselect("Select columns for analysis:", df.columns)

    if selected_columns:
        for column in selected_columns:
            st.subheader(f"Bar Chart for {column}")
            st.bar_chart(df[column].value_counts())

            st.subheader(f"Line Chart for {column}")
            st.line_chart(df[[column]])

            st.subheader(f"Box Plot for {column}")
            if df[column].dtype in ['int64', 'float64']:
                sns.set(style="whitegrid")
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=df[column])
                st.pyplot(plt.gcf())
            else:
                st.warning(f"Cannot create box plot for non-numeric column: {column}")

            st.subheader(f"Scatter Plot for {column}")
            if df[column].dtype in ['int64', 'float64']:
                st.scatter_chart(df[[column]])

            st.subheader(f"Area Chart for {column}")
            if df[column].dtype in ['int64', 'float64']:
                st.area_chart(df[[column]])

def advanced_analytics(df):
    st.subheader("Advanced Analytics:")

    # Clustering
    st.write("### K-Means Clustering:")
    st.info("Performing K-Means clustering on numeric columns.")

    numeric_df = df.select_dtypes(include=['number'])

    # Handle missing values before clustering
    imputer = SimpleImputer(strategy='mean')
    numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

    # Check for NaN after imputation
    if numeric_df_imputed.isnull().sum().sum() > 0:
        st.warning("Some missing values could not be imputed. Consider dropping or imputing them manually.")
        st.stop()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df_imputed)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    st.write("#### Cluster Distribution:")
    st.bar_chart(df['Cluster'].value_counts())

    # Principal Component Analysis (PCA)
    st.write("### Principal Component Analysis (PCA):")
    st.info("Performing PCA on numeric columns for dimensionality reduction.")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_pca = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])
    st.scatter_chart(df_pca)




from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def handle_categorical_data(df):
    st.subheader("Handling Categorical Data:")

    categorical_columns = df.select_dtypes(['object', 'category']).columns

    if categorical_columns.empty:
        st.info("No categorical columns found.")
        return df

    df_encoded = df.copy()

    for column in categorical_columns:
        unique_values_count = df[column].nunique()

        # Choose between one-hot encoding and label encoding based on unique values count
        if unique_values_count > 10:
            st.info(f"Applying One-Hot Encoding to '{column}' (nominal data).")
            onehot_encoder = OneHotEncoder(drop='first')
            encoded_data = onehot_encoder.fit_transform(df[[column]]).toarray()
            
            # Construct feature names manually
            feature_names = onehot_encoder.get_feature_names_out([column])
            encoded_data = pd.DataFrame(encoded_data, columns=feature_names)
            
            df_encoded = pd.concat([df_encoded, encoded_data], axis=1)
            df_encoded = df_encoded.drop(column, axis=1)  # Drop original categorical column
        else:
            st.info(f"Applying Label Encoding to '{column}' (ordinal data).")
            label_encoder = LabelEncoder()
            df_encoded[column] = label_encoder.fit_transform(df[column])

    st.success("Categorical Data Handling Completed.")
    return df_encoded


from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(df, columns_to_normalize=None):
    st.subheader("Feature Scaling - Min-Max Normalization:")

    if not columns_to_normalize:
        columns_to_normalize = df.select_dtypes(['int64', 'float64']).columns

    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    st.success("Data Normalized Successfully.")
    return df_normalized

def standardize_data(df, columns_to_standardize=None):
    st.subheader("Feature Scaling - Standardization:")

    if not columns_to_standardize:
        columns_to_standardize = df.select_dtypes(['int64', 'float64']).columns

    scaler = StandardScaler()
    df_standardized = df.copy()
    df_standardized[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    st.success("Data Standardized Successfully.")
    return df_standardized

######################################################################################################
######################################################################################################
import pandas as pd
from sklearn.impute import SimpleImputer
import streamlit as st

def handle_missing_values(df):
    """
    Handle missing values in a DataFrame.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame with missing values.

    Returns:
    - df_imputed: pd.DataFrame
        DataFrame with missing values imputed.
    """
    st.subheader("Handling Missing Values:")

    # Identify columns with missing values
    columns_with_missing_values = df.columns[df.isnull().any()]

    if columns_with_missing_values.empty:
        st.success("No missing values found.")
        return df

    # Iterate through columns with missing values and impute based on characteristics
    for col in columns_with_missing_values:
        df[col] = impute_column(df[col])

    st.success("Missing values imputed successfully.")
    return df

def impute_column(column):
    """
    Impute missing values in a column based on column characteristics.

    Parameters:
    - column: pd.Series
        The column with missing values.

    Returns:
    - column_imputed: pd.Series
        Column with missing values imputed.
    """
    imputation_strategy = determine_imputation_strategy(column)
    imputer = SimpleImputer(strategy=imputation_strategy)
    column_imputed = pd.Series(imputer.fit_transform(column.values.reshape(-1, 1)).ravel(), index=column.index)
    return column_imputed

def determine_imputation_strategy(column):
    """
    Determine the appropriate imputation strategy based on column characteristics.

    Parameters:
    - column: pd.Series
        The column with missing values.

    Returns:
    - imputation_strategy: str
        Imputation strategy: 'mean', 'median', or 'mode'.
    """
    if column.dtype == 'object':
        # Categorical column: Use mode imputation
        return 'most_frequent'
    elif abs(column.skew()) > 1:
        # Skewed numeric column: Use median imputation
        return 'median'
    else:
        # Approximately normally distributed numeric column: Use mean imputation
        return 'mean'
########################################################################################
    ############################################################################



def download_processed_data(df):
    # Convert the DataFrame to CSV
    csv_data = df.to_csv(index=False)

    # Encoding the CSV data with base64
    csv_data_encoded = base64.b64encode(csv_data.encode()).decode()

    # Create a download link for the CSV file
    href = f'<a href="data:file/csv;base64,{csv_data_encoded}" download="processed_data.csv">Download Processed Data</a>'

    # Display the download link
    st.markdown(href, unsafe_allow_html=True)

###################################################################################################
###################################################################################################
def handle_duplicates(df):
    """
    Handle duplicate rows in a DataFrame.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame.

    Returns:
    - df_no_duplicates: pd.DataFrame
        DataFrame with duplicates removed.
    """
    st.subheader("Handling Duplicates:")

    # Check for duplicate rows
    num_duplicates = df.duplicated().sum()

    if num_duplicates == 0:
        st.success("No duplicate rows found.")
        return df

    # Remove duplicate rows
    df_no_duplicates = df.drop_duplicates()

    st.warning(f"Removed {num_duplicates} duplicate rows.")
    return df_no_duplicates

#############################################################################################
##############################################################################################



if __name__ == "__main__":
    st.title("Deloitte Data Analysis Tool")
    st.header("Data analysis application developed by Emmanuel for PA Team")

    uploaded_file = st.file_uploader("Upload your data (CSV or Excel)...", type=["csv", "xlsx"],
                                     help="Please upload data (CSV or Excel)")

    if uploaded_file:
        df = load_and_display_data(uploaded_file)
        # df_balanced = handle_imbalanced_data(df)
        display_basic_info(df)
        exploratory_data_analysis(df)
        df_encoded = handle_categorical_data(df)
        df_normalized = normalize_data(df_encoded)
        df_standardized = standardize_data(df_normalized)
        # df_balanced = handle_imbalanced_data(df_standardized)
        display_basic_info(df_standardized)
        data_analysis_charts(df_standardized)
        advanced_analytics(df_standardized)
        download_processed_data(df_standardized)
    else:
        st.warning("Please upload a CSV or Excel file to perform data analysis.")
