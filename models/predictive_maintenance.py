from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def feature_importance_analysis(df, target_col, feature_cols):
    """
    Entrena un Random Forest para descubrir qué sensores 
    tienen el mayor impacto (correlación no lineal) sobre una variable objetivo.
    Ejemplo: ¿Qué causa que la EGT (Exhaust Temp) suba?
    """
    if df.empty or target_col not in df.columns or not all(f in df.columns for f in feature_cols):
        return pd.DataFrame()
        
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Random Forest Regressor (apto para CPU/GPU)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Extraer importancia de cada sensor
    importance = pd.DataFrame({
        'Sensor': feature_cols,
        'Importancia': rf.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    return importance

def cluster_operating_states(df, features, n_clusters=3):
    """
    Agrupa los datos históricos usando K-Means para encontrar 
    automáticamente estados operacionales ocultos.
    """
    if df.empty or not all(f in df.columns for f in features):
        return df
        
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df_clean
