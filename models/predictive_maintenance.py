from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
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

def predict_rul_linear(df, target_col, threshold=90.0, mode='increasing'):
    """
    Realiza una regresión lineal sobre el tiempo (en segundos) para predecir 
    cuándo la variable target_col cruzará el umbral (threshold) especificado.
    mode: 'increasing' (la variable sube hacia el umbral, ej. obstrucción de filtro)
          'decreasing' (la variable baja hacia el umbral, ej. delta de intercambiador)
    Devuelve: (current_val, slope_per_hour, hours_to_threshold)
    """
    if df.empty or target_col not in df.columns or len(df) < 5:
        return None, None, None
        
    df_clean = df.dropna(subset=[target_col, 'timestamp']).copy()
    if len(df_clean) < 5:
        return None, None, None
        
    # Convertir timestamp a segundos
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    df_clean['seconds'] = (df_clean['timestamp'] - df_clean['timestamp'].min()).dt.total_seconds()
    
    X = df_clean[['seconds']].values
    y = df_clean[target_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    current_val = y[-1]
    slope = model.coef_[0]
    slope_per_hour = slope * 3600
    
    if mode == 'increasing':
        if slope <= 0:
            hours_to_threshold = float('inf')
        else:
            diff = threshold - current_val
            if diff <= 0:
                hours_to_threshold = 0.0
            else:
                hours_to_threshold = (diff / slope) / 3600
    else:  # decreasing
        if slope >= 0:
            hours_to_threshold = float('inf')
        else:
            diff = current_val - threshold
            if diff <= 0:
                hours_to_threshold = 0.0
            else:
                hours_to_threshold = (diff / -slope) / 3600
                
    return current_val, slope_per_hour, hours_to_threshold
