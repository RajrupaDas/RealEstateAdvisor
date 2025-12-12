import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn 
import mlflow.pyfunc 

mlflow.set_tracking_uri("file:./mlruns")

CLASSIFICATION_MODEL_URI = "models:/Best_Classification_Model/2"
REGRESSION_MODEL_URI = "models:/Best_Regression_Model/2"

@st.cache_resource
def load_models():
    """Loads the registered models from MLflow using the appropriate flavor."""
    try:
        cls_model = mlflow.sklearn.load_model(CLASSIFICATION_MODEL_URI) 
        
        reg_model = mlflow.pyfunc.load_model(REGRESSION_MODEL_URI)
        
        st.success("XGBoost Models (Version 2) loaded successfully from MLflow!")
        return cls_model, reg_model
    except Exception as e:
        st.error(f"Error loading models from MLflow. Ensure 'mlruns' folder exists and models are registered. Check if Version 2 exists. Error: {e}")
        st.info("If the error persists, try changing the version back to '1' or check the MLflow UI.")
        return None, None


cls_pipeline, reg_pipeline = load_models()


try:
    df_meta = pd.read_csv('processed_data.csv')
except FileNotFoundError:
    st.error("processed_data.csv not found. Please run data_preprocessing.py first.")
    df_meta = pd.DataFrame() 

def get_unique_sorted_values(column):
    """Safely retrieves unique sorted values for select boxes."""
    if column in df_meta.columns:
        valid_values = df_meta[column].dropna().unique()
        return sorted([str(val) for val in valid_values])
    return []


st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

st.title("Real Estate Investment Advisor (with XGBoost)")
st.markdown("---")

if cls_pipeline is None or reg_pipeline is None:
    st.stop()


st.sidebar.header("Property Details Input")
st.sidebar.markdown("Enter the specifications to get an investment forecast.")

with st.sidebar.form("input_form"):
    
    
    city = st.selectbox("City", get_unique_sorted_values('City'))
    locality = st.selectbox("Locality", get_unique_sorted_values('Locality'))
    property_type = st.selectbox("Property Type", get_unique_sorted_values('Property_Type'))
    furnished_status = st.selectbox("Furnished Status", get_unique_sorted_values('Furnished_Status'))
    parking_space = st.selectbox("Parking Space", get_unique_sorted_values('Parking_Space'))
    security = st.selectbox("Security Features", get_unique_sorted_values('Security'))
    amenities = st.selectbox("Amenities", get_unique_sorted_values('Amenities'))
    
    
    bhk = st.slider("BHK (Bedrooms, Hall, Kitchen)", 1, 6, 3)
    size_in_sqft = st.number_input("Size in SqFt", min_value=100, value=1500, step=50)
    floor_no = st.number_input("Floor Number", min_value=0, value=1)
    total_floors = st.number_input("Total Floors in Building", min_value=1, value=10)
    age_of_property = st.slider("Age of Property (Years)", 0, 50, 5)
    nearby_schools = st.number_input("Nearby Schools Score/Count", min_value=0, value=2)
    nearby_hospitals = st.number_input("Nearby Hospitals Score/Count", min_value=0, value=2)
    
    submit_button = st.form_submit_button("Get Investment Recommendation")


if submit_button:
    
    
    input_data = {
        'State': get_unique_sorted_values('State')[0] if get_unique_sorted_values('State') else 'N/A',
        'City': city,
        'Locality': locality,
        'Property_Type': property_type,
        'BHK': bhk,
        'Size_in_SqFt': size_in_sqft,
        'Furnished_Status': furnished_status,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age_of_property,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Parking_Space': parking_space,
        'Security': security,
        'Amenities': amenities,
        'Facing': get_unique_sorted_values('Facing')[0] if get_unique_sorted_values('Facing') else 'N/A',
        'Owner_Type': get_unique_sorted_values('Owner_Type')[0] if get_unique_sorted_values('Owner_Type') else 'N/A',
        'Availability_Status': get_unique_sorted_values('Availability_Status')[0] if get_unique_sorted_values('Availability_Status') else 'N/A',
        'Public_Transport_Accessibility': get_unique_sorted_values('Public_Transport_Accessibility')[0] if get_unique_sorted_values('Public_Transport_Accessibility') else 'N/A'
    }

    input_df = pd.DataFrame([input_data])
    
    with st.spinner('Calculating forecasts and analyzing investment potential...'):
        
        
        cls_prediction = cls_pipeline.predict(input_df)[0]
        cls_proba = cls_pipeline.predict_proba(input_df)[0][1] 

       
        reg_prediction = reg_pipeline.predict(input_df)[0]
    
    
    st.header("Investment Forecast & Recommendation")
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification: Potential Assessment")
        
        if cls_prediction == 1:
            st.success(f"**YES**, this property is flagged as a Good Investment!")
            st.markdown(f"**Confidence Score:** **{cls_proba:.1%}** (Likelihood of positive appreciation)")
            st.balloons()
        else:
            st.warning(f"**CAUTION**, investment potential is low.")
            st.markdown(f"**Confidence Score:** **{cls_proba:.1%}** (Likelihood of positive appreciation)")

    with col2:
        st.subheader("Regression: Estimated Price after 5 Years")
        st.metric(label="Predicted Price (5 Years)", value=f"₹ {reg_prediction:,.2f} Lakhs")
        
        
        st.info("**Prediction Reliability:** The XGBoost model provides a high-fidelity price forecast.")

    
    st.markdown("---")
    st.header("Data Visualizations (EDA)")
    
    
    st.subheader("Price Distribution in Selected City")
    if not df_meta.empty and city in df_meta['City'].unique():
        city_data = df_meta[df_meta['City'] == city]['Price_in_Lakhs']
        st.bar_chart(city_data, use_container_width=True)
