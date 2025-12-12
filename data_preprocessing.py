import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def create_target_variables(df, assumed_annual_growth_rate=0.08):
 
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1 + assumed_annual_growth_rate) ** 5

 
    df['Price_per_SqFt'] = pd.to_numeric(df['Price_per_SqFt'], errors='coerce')
    
    city_median_price_sqft = df.groupby('City')['Price_per_SqFt'].median().reset_index()
    city_median_price_sqft.rename(columns={'Price_per_SqFt': 'Median_City_Price_SqFt'}, inplace=True)
    df = df.merge(city_median_price_sqft, on='City', how='left')
    
    df.dropna(subset=['Price_per_SqFt', 'Median_City_Price_SqFt'], inplace=True)
    
    df['Good_Investment'] = (df['Price_per_SqFt'] < df['Median_City_Price_SqFt']).astype(int)
    df.drop(columns=['Median_City_Price_SqFt'], inplace=True)
    
    return df

def get_preprocessor(numerical_features, categorical_features):
 
    from sklearn.impute import SimpleImputer
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),         ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

if __name__ == '__main__':
    data_path = 'india_housing_prices.csv'
    
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please place the file in the project directory.")
        exit()

    current_year = pd.Timestamp('now').year
    df['Year_Built'] = pd.to_numeric(df['Year_Built'], errors='coerce').fillna(current_year).astype(int)
    df['Age_of_Property'] = current_year - df['Year_Built']
    
    df = create_target_variables(df)
       NUM_COLS_TO_CONVERT = ['BHK', 'Size_in_SqFt', 'Nearby_Schools', 'Nearby_Hospitals', 'Total_Floors', 'Floor_No']
    for col in NUM_COLS_TO_CONVERT:
               df[col] = pd.to_numeric(df[col], errors='coerce')

       df.drop(columns=['ID', 'Year_Built', 'Price_per_SqFt'], inplace=True)
    numerical_cols = ['BHK', 'Size_in_SqFt', 'Age_of_Property', 'Nearby_Schools', 
                      'Nearby_Hospitals', 'Total_Floors', 'Floor_No']
    
    categorical_cols = ['State', 'City', 'Locality', 'Property_Type', 'Furnished_Status', 
                        'Public_Transport_Accessibility', 'Security', 'Amenities', 'Facing', 
                        'Owner_Type', 'Availability_Status', 'Parking_Space'] # Moved Parking_Space here
    
       X = df.drop(columns=['Future_Price_5Y', 'Good_Investment', 'Price_in_Lakhs'])
    y_reg = df['Future_Price_5Y']
    y_cls = df['Good_Investment']
    
    numerical_features = [col for col in numerical_cols if col in X.columns]
    categorical_features = [col for col in categorical_cols if col in X.columns]
    
    df_clean = X.copy()
    df_clean['Future_Price_5Y'] = y_reg
    df_clean['Good_Investment'] = y_cls
    df_clean['Price_in_Lakhs'] = df['Price_in_Lakhs']
    df_clean.to_csv('processed_data.csv', index=False)
    
    print("Data Preprocessing Complete. File 'processed_data.csv' created.")
    print(f"Regression Target ('Future_Price_5Y') created.")
    print(f"Classification Target ('Good_Investment') created.")
