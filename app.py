import streamlit as st
import pandas as pd
import plotly.express as px
from data_extractor import fetch_telemetry_data
from models.anomaly_detection import detect_anomalies
from models.predictive_maintenance import feature_importance_analysis, cluster_operating_states
from report_generator import SENSOR_NAMES

st.set_page_config(page_title="AI Marine Analyzer", page_icon="🛥️", layout="wide")

st.title("🧠 AI Marine Analyzer - CAT 3516B")
st.markdown("Plataforma local de Machine Learning acelerada por hardware para el análisis de telemetría masiva de InfluxDB.")

# --- SIDEBAR: Configuración de Datos ---
st.sidebar.header("📡 1. Extracción de Datos")
days = st.sidebar.slider("Ventana Histórica (Días)", min_value=1, max_value=60, value=30)

if st.sidebar.button("Cargar Datos de InfluxDB"):
    with st.spinner(f"Descargando datos de los últimos {days} días (agrupados por minuto)..."):
        df = fetch_telemetry_data(days_back=days)
        if not df.empty:
            st.session_state['df'] = df
            st.sidebar.success(f"¡{len(df)} registros descargados!")
        else:
            st.sidebar.error("No se encontraron datos.")

# Verificar si hay datos cargados en memoria
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # --- PESTAÑAS DE ANÁLISIS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Explorador", "🚨 Anomalías", "🔮 Predictivo", "📄 Informe Ejecutivo"])
    
    with tab1:
        st.subheader("Vista Preliminar de la Telemetría")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Gráfico rápido
        if "engine_rpm" in df.columns:
            fig = px.line(df, x="timestamp", y="engine_rpm", title="Historial de RPM del Motor")
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.subheader("Isolation Forest: Detección Multivariable")
        st.markdown("Este algoritmo busca momentos donde el motor se comportó de forma anormal (ej. alta temperatura con bajas RPM).")
        
        features_to_check = st.multiselect("Selecciona sensores para analizar anomalías conjuntas:", 
                                           options=[c for c in df.columns if c not in ['timestamp']],
                                           default=['engine_rpm', 'oil_pressure', 'engine_temperature'])
        
        contamination = st.slider("Sensibilidad (Estimación % de fallas)", 0.01, 0.20, 0.05)
        
        if st.button("Ejecutar Isolation Forest"):
            with st.spinner("Entrenando modelo de detección..."):
                df_anomalies = detect_anomalies(df, features_to_check, contamination)
                
                # Visualizar anomalías
                if not df_anomalies.empty and 'is_anomaly' in df_anomalies.columns:
                    anom_count = df_anomalies['is_anomaly'].sum()
                    st.session_state['anomalies_count'] = anom_count
                    
                    col1 = features_to_check[0] if len(features_to_check) > 0 else df.columns[1]
                    fig_anom = px.scatter(df_anomalies, x="timestamp", y=col1, color="is_anomaly", 
                                          color_discrete_map={False: '#1f77b4', True: '#ef4444'},
                                          title=f"Detección de Anomalías basadas en {len(features_to_check)} sensores")
                    st.plotly_chart(fig_anom, use_container_width=True)
                    
                    st.error(f"Se detectaron {anom_count} eventos anómalos.")
                    
    with tab3:
        st.subheader("Random Forest: Análisis de Causa Raíz (Feature Importance)")
        st.markdown("Descubre qué sensores influyen más en el aumento de una variable crítica.")
        
        target = st.selectbox("Selecciona la Variable Crítica a analizar (ej. Temperatura Escape):", 
                              options=[c for c in df.columns if c not in ['timestamp']],
                              index=list(df.columns).index('exhaust_temperature') if 'exhaust_temperature' in df.columns else 0)
                              
        if st.button("Ejecutar Random Forest (Análisis de Importancia)"):
            with st.spinner("Entrenando ensamble de árboles de decisión (Random Forest)..."):
                features = [c for c in df.columns if c not in ['timestamp', target, 'is_anomaly', 'anomaly_score', 'cluster']]
                importance_df = feature_importance_analysis(df, target_col=target, feature_cols=features)
                
                if not importance_df.empty:
                    st.session_state['importance_df'] = importance_df
                    fig_rf = px.bar(importance_df, x='Importancia', y='Sensor', orientation='h', 
                                    title=f"Impacto de los sensores sobre {target}")
                    st.plotly_chart(fig_rf, use_container_width=True)
                    
    with tab4:
        st.subheader("📄 Informe Técnico Consolidado")
        st.markdown("Revisión automática generada por los modelos de Inteligencia Artificial.")
        
        start_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M') if 'timestamp' in df.columns else 'N/A'
        end_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M') if 'timestamp' in df.columns else 'N/A'
        total_records = len(df)
        anomalies_count = st.session_state.get('anomalies_count', 0)
        imp_df = st.session_state.get('importance_df', None)
        
        # Renderizar en pantalla (Markdown)
        informe_md = f"""
### 1. Resumen Operacional
*   **Período Analizado:** {start_date} al {end_date}
*   **Total de Registros Evaluados:** {total_records}

### 2. Diagnóstico de Integridad (Isolation Forest)
*   **Anomalías Severas Detectadas:** {anomalies_count}
"""
        if anomalies_count > 0: informe_md += "*   🚨 **ATENCIÓN:** Se requiere inspección mecánica de los eventos reportados.\n"
            
        informe_md += "\n### 3. Análisis de Causa Raíz (Random Forest)\n"
        if imp_df is not None and not imp_df.empty:
            informe_md += "Parámetros físicos con mayor impacto en fallas críticas:\n"
            for i, row in imp_df.head(3).iterrows():
                sensor_clean = SENSOR_NAMES.get(row['Sensor'], row['Sensor'])
                informe_md += f"*   **{sensor_clean}**: {row['Importancia']*100:.1f}% de impacto\n"
        else:
            informe_md += "*   *Aún no se ha ejecutado el análisis de causa raíz. Ve a la pestaña Predictivo para generarlo.*\n"
            
        st.info(informe_md)
        st.markdown("---")
        
        # Botón de Descarga
        try:
            from report_generator import generate_pdf
            pdf_bytes = generate_pdf(df, anomalies_count, imp_df)
            st.download_button(
                label="📥 Descargar Informe Detallado en PDF",
                data=pdf_bytes,
                file_name=f"Reporte_Tecnico_Avanzado_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except ImportError:
            st.error("⚠️ Falta la librería de PDF. Ejecuta `pip install fpdf` en tu terminal para habilitar la descarga.")
else:
    st.info("👈 Utiliza el panel izquierdo para descargar los datos desde InfluxDB y comenzar el análisis.")
