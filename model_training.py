import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline

from data_preprocessing import get_preprocessor 
import mlflow
import mlflow.sklearn
import numpy as np
from xgboost import XGBClassifier, XGBRegressor


mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("Real_Estate_Investment_Advisor")

def train_and_log_model(X, y, task_type, model, model_name, preprocessor):
    
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
  
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model)])
    
        with mlflow.start_run(run_name=f"{model_name}_{task_type}_XGBoost"):
        
     
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)

        if task_type == 'Classification':
            try:
                y_proba = full_pipeline.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except AttributeError:
                roc_auc = 0.0

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc
            }
            print(f"Classification Metrics (XGBoost): {metrics}")
            
        elif task_type == 'Regression':
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': r2_score(y_test, y_pred)
            }
            print(f"Regression Metrics (XGBoost): {metrics}")
            
        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(
            full_pipeline, 
            "model", 
            registered_model_name=f"Best_{task_type}_Model"
        )

        print(f"MLflow Run completed for {model_name} (XGBoost {task_type}).")
        
        return full_pipeline

if __name__ == '__main__':
    df = pd.read_csv('processed_data.csv')
    
    X = df.drop(columns=['Future_Price_5Y', 'Good_Investment', 'Price_in_Lakhs'])
    y_reg = df['Future_Price_5Y']
    y_cls = df['Good_Investment']

    numerical_cols = ['BHK', 'Size_in_SqFt', 'Age_of_Property', 'Nearby_Schools', 
                      'Nearby_Hospitals', 'Total_Floors', 'Floor_No']
    categorical_cols = ['State', 'City', 'Locality', 'Property_Type', 'Furnished_Status', 
                        'Public_Transport_Accessibility', 'Security', 'Amenities', 'Facing', 
                        'Owner_Type', 'Availability_Status', 'Parking_Space']
    
    numerical_features = [col for col in numerical_cols if col in X.columns]
    categorical_features = [col for col in categorical_cols if col in X.columns]
    
    preprocessor = get_preprocessor(numerical_features, categorical_features)

   
    cls_model_xgb = XGBClassifier(
        objective='binary:logistic', 
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42
    )
    best_cls_model_xgb = train_and_log_model(X, y_cls, 'Classification', cls_model_xgb, 'XGBoostClassifier', preprocessor)

    
    reg_model_xgb = XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42
    )
    best_reg_model_xgb = train_and_log_model(X, y_reg, 'Regression', reg_model_xgb, 'XGBoostRegressor', preprocessor)

    print("\nModel training with XGBoost complete. Check MLflow UI for new metrics.")
