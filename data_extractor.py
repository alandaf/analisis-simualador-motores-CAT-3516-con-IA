import pandas as pd
from influxdb_client import InfluxDBClient
import streamlit as st

# Configuración estática de InfluxDB (basada en el simulador)
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "TOKEN_REVOCADO"
INFLUXDB_ORG = "Motor Data"
INFLUXDB_BUCKET = "monitoreoram"

@st.cache_data(ttl=3600)  # Caché de 1 hora para no saturar InfluxDB con peticiones repetidas
def fetch_telemetry_data(days_back=30):
    """
    Descarga el historial de telemetría de InfluxDB.
    Para una ventana de 30 días, agrupa los datos en promedios de 1 minuto 
    (downsampling) para evitar colapsar la RAM de la laptop.
    """
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()
    
    # Consulta Flux optimizada (agrupando promedios por minuto)
    query = f'''
    from(bucket:"{INFLUXDB_BUCKET}") 
    |> range(start: -{days_back}d) 
    |> filter(fn: (r) => r["_measurement"] == "motor_data") 
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    try:
        # Devuelve directamente un DataFrame de Pandas
        df = query_api.query_data_frame(query, org=INFLUXDB_ORG)
        if df.empty:
            return pd.DataFrame()
        
        # Limpieza de DataFrame
        if isinstance(df, list):
            df = df[0] # En caso de que InfluxDB devuelva una lista de DataFrames
            
        # Eliminar columnas internas de InfluxDB
        cols_to_drop = ["result", "table", "_start", "_stop", "_measurement"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # Renombrar "_time" a "timestamp"
        if "_time" in df.columns:
            df = df.rename(columns={"_time": "timestamp"})
            
        # Rellenar nulos (Forward Fill)
        df = df.ffill()
        
        return df
    except Exception as e:
        st.error(f"Error de conexión a InfluxDB: {e}")
        return pd.DataFrame()

def fetch_maintenance_data(days_back=30):
    """
    Descarga el historial de eventos de mantenimiento de InfluxDB.
    """
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()
    
    query = f'''
    from(bucket:"{INFLUXDB_BUCKET}") 
    |> range(start: -{days_back}d) 
    |> filter(fn: (r) => r["_measurement"] == "maintenance_events") 
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        df = query_api.query_data_frame(query, org=INFLUXDB_ORG)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df, list):
            df = df[0]
            
        cols_to_drop = ["result", "table", "_start", "_stop", "_measurement"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        if "_time" in df.columns:
            df = df.rename(columns={"_time": "timestamp"})
            
        return df
    except Exception as e:
        print(f"Error fetching maintenance from InfluxDB: {e}")
        return pd.DataFrame()

