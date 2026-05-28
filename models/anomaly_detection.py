from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def detect_anomalies(df, features, contamination=0.05):
    """
    Usa Isolation Forest para detectar anomalías en la telemetría.
    contamination: Porcentaje estimado de datos anómalos.
    """
    if df.empty or not all(f in df.columns for f in features):
        return df
        
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]
    
    # Estandarizar variables (es vital para IA multivariable)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar modelo
    # n_jobs=-1 usa todos los núcleos del CPU disponibles en la laptop
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    df_clean['anomaly_score'] = model.fit_predict(X_scaled)
    
    # IsolationForest devuelve -1 para anomalías y 1 para datos normales
    df_clean['is_anomaly'] = df_clean['anomaly_score'] == -1
    
    return df_clean
