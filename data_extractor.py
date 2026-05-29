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
    Descarga el historial de telemetría de InfluxDB, incluyendo datos de motor,
    gps y cilindros. Agrupa por 1 minuto (downsampling).
    """
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()
    
    # 1. Consulta Flux para motor_data
    query_motor = f'''
    from(bucket:"{INFLUXDB_BUCKET}") 
    |> range(start: -{days_back}d) 
    |> filter(fn: (r) => r["_measurement"] == "motor_data") 
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    # 2. Consulta Flux para gps_data
    query_gps = f'''
    from(bucket:"{INFLUXDB_BUCKET}") 
    |> range(start: -{days_back}d) 
    |> filter(fn: (r) => r["_measurement"] == "gps_data") 
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    # 3. Consulta Flux para cylinder_data
    query_cyl = f'''
    from(bucket:"{INFLUXDB_BUCKET}") 
    |> range(start: -{days_back}d) 
    |> filter(fn: (r) => r["_measurement"] == "cylinder_data") 
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    def clean_df(df_raw):
        if df_raw.empty:
            return pd.DataFrame()
        if isinstance(df_raw, list):
            df_raw = df_raw[0]
        # Eliminar columnas internas de InfluxDB
        cols_to_drop = ["result", "table", "_start", "_stop", "_measurement"]
        df_raw = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns], errors='ignore')
        # Renombrar "_time" a "timestamp"
        if "_time" in df_raw.columns:
            df_raw = df_raw.rename(columns={"_time": "timestamp"})
        return df_raw
    
    try:
        # Cargar motor_data
        df_motor = query_api.query_data_frame(query_motor, org=INFLUXDB_ORG)
        df_motor = clean_df(df_motor)
        
        if df_motor.empty:
            return pd.DataFrame()
        
        # Intentar obtener GPS
        try:
            df_gps = query_api.query_data_frame(query_gps, org=INFLUXDB_ORG)
            df_gps = clean_df(df_gps)
        except Exception as eg:
            print(f"Error querying gps_data: {eg}")
            df_gps = pd.DataFrame()
            
        # Intentar obtener cilindros
        try:
            df_cyl = query_api.query_data_frame(query_cyl, org=INFLUXDB_ORG)
            df_cyl = clean_df(df_cyl)
        except Exception as ec:
            print(f"Error querying cylinder_data: {ec}")
            df_cyl = pd.DataFrame()
            
        # Unificar usando merge_asof en timestamp
        df_merged = df_motor
        df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
        df_merged = df_merged.sort_values('timestamp')
        
        if not df_gps.empty:
            df_gps['timestamp'] = pd.to_datetime(df_gps['timestamp'])
            df_gps = df_gps.sort_values('timestamp')
            df_merged = pd.merge_asof(df_merged, df_gps, on="timestamp", direction="nearest")
            
        if not df_cyl.empty:
            df_cyl['timestamp'] = pd.to_datetime(df_cyl['timestamp'])
            df_cyl = df_cyl.sort_values('timestamp')
            df_merged = pd.merge_asof(df_merged, df_cyl, on="timestamp", direction="nearest")
            
        # Rellenar nulos (Forward Fill y Backward Fill)
        df_merged = df_merged.ffill().bfill()
        
        return df_merged
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

