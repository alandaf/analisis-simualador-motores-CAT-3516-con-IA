import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from data_extractor import fetch_telemetry_data, fetch_maintenance_data
from models.anomaly_detection import detect_anomalies
from models.predictive_maintenance import feature_importance_analysis, cluster_operating_states, predict_rul_linear
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
        df_maint = fetch_maintenance_data(days_back=days)
        if not df.empty:
            st.session_state['df'] = df
            st.session_state['df_maint'] = df_maint
            st.sidebar.success(f"¡{len(df)} registros descargados!")
        else:
            st.sidebar.error("No se encontraron datos.")

# Verificar si hay datos cargados en memoria
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # --- PESTAÑAS DE ANÁLISIS ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Explorador", 
        "🚨 Anomalías", 
        "🔮 Predictivo & RUL", 
        "🔄 Clústeres", 
        "⚙️ Balance V16", 
        "🌱 ECO & SCR", 
        "🗺️ GPS & Ruta", 
        "📄 Informe & Manual"
    ])
    
    with tab1:
        st.subheader("Vista Preliminar de la Telemetría")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Gráficos rápidos en columnas
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if "engine_rpm" in df.columns:
                fig = px.line(df, x="timestamp", y="engine_rpm", title="Historial de RPM del Motor")
                st.plotly_chart(fig, use_container_width=True)
        with col_g2:
            if "engine_temperature" in df.columns:
                fig_t = px.line(df, x="timestamp", y="engine_temperature", title="Historial de Temperatura del Refrigerante")
                st.plotly_chart(fig_t, use_container_width=True)
                
        # Heatmap de correlación de Pearson
        st.subheader("Correlation Heatmap (Mapa de Calor de Correlaciones)")
        numeric_cols = [c for c in df.columns if c not in ['timestamp', 'is_anomaly', 'anomaly_score', 'cluster', 'latitude', 'longitude'] and not c.startswith('cylinder_')]
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", 
                                 title="Correlación Lineal entre Sensores (Pearson)",
                                 color_continuous_scale='RdBu_r', origin='lower')
            st.plotly_chart(fig_corr, use_container_width=True)
            
    with tab2:
        st.subheader("Isolation Forest: Detección Multivariable")
        st.markdown("Este algoritmo busca momentos donde el motor se comportó de forma anormal (ej. alta temperatura con bajas RPM).")
        
        features_to_check = st.multiselect("Selecciona sensores para analizar anomalías conjuntas:", 
                                           options=[c for c in df.columns if c not in ['timestamp', 'latitude', 'longitude'] and not c.startswith('cylinder_')],
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
                              options=[c for c in df.columns if c not in ['timestamp', 'latitude', 'longitude'] and not c.startswith('cylinder_')],
                              index=list(df.columns).index('exhaust_temperature') if 'exhaust_temperature' in df.columns else 0)
                              
        if st.button("Ejecutar Random Forest (Análisis de Importancia)"):
            with st.spinner("Entrenando ensamble de árboles de decisión (Random Forest)..."):
                features = [c for c in df.columns if c not in ['timestamp', target, 'is_anomaly', 'anomaly_score', 'cluster', 'latitude', 'longitude'] and not c.startswith('cylinder_')]
                importance_df = feature_importance_analysis(df, target_col=target, feature_cols=features)
                
                if not importance_df.empty:
                    st.session_state['importance_df'] = importance_df
                    fig_rf = px.bar(importance_df, x='Importancia', y='Sensor', orientation='h', 
                                    title=f"Impacto de los sensores sobre {target}")
                    st.plotly_chart(fig_rf, use_container_width=True)
                    
        # --- SECCIÓN: Mantenimiento Predictivo RUL ---
        st.markdown("---")
        st.subheader("🔮 Estimación de Vida Útil Restante (RUL - Regresión Lineal)")
        st.markdown("Modelado predictivo de series de tiempo para anticipar el mantenimiento preventivo de componentes críticos.")
        
        col_rul1, col_rul2 = st.columns(2)
        with col_rul1:
            st.markdown("#### 🛢️ Degradación del Filtro de Aceite")
            if 'filter_clog' in df.columns:
                cur_val, slope, hours_left = predict_rul_linear(df, 'filter_clog', threshold=90.0, mode='increasing')
                if cur_val is not None:
                    st.metric("Obstrucción Actual", f"{cur_val:.1f}%", delta=f"{slope:.4f}% / hora", delta_color="inverse")
                    if hours_left == float('inf'):
                        st.success("✅ Filtro estable: Sin tendencia de obstrucción progresiva detectada.")
                    else:
                        st.warning(f"⚠️ Reemplazo recomendado en aproximadamente **{hours_left:.1f} horas** de operación (Umbral: 90.0%).")
                        
                        df_proj = df[['timestamp', 'filter_clog']].copy()
                        df_proj['timestamp'] = pd.to_datetime(df_proj['timestamp'])
                        last_time = df_proj['timestamp'].max()
                        future_times = [last_time + pd.Timedelta(hours=h) for h in np.linspace(0, max(1.0, hours_left), 50)]
                        future_vals = [cur_val + (slope * h) for h in np.linspace(0, max(1.0, hours_left), 50)]
                        df_future = pd.DataFrame({'timestamp': future_times, 'filter_clog': future_vals, 'Tipo': 'Proyección (IA)'})
                        df_proj['Tipo'] = 'Histórico'
                        df_comb = pd.concat([df_proj, df_future])
                        
                        fig_proj = px.line(df_comb, x='timestamp', y='filter_clog', color='Tipo', 
                                           title='Proyección de Obstrucción del Filtro de Aceite',
                                           color_discrete_map={'Histórico': '#1f77b4', 'Proyección (IA)': '#ef4444'})
                        fig_proj.add_hline(y=90.0, line_dash="dash", line_color="red", annotation_text="Límite Crítico (90%)")
                        st.plotly_chart(fig_proj, use_container_width=True)
            else:
                st.info("No hay datos de obstrucción de filtro disponibles.")
                
        with col_rul2:
            st.markdown("#### 🥶 Eficiencia del Intercambiador de Calor")
            if 'heat_exchanger_delta_t' in df.columns:
                cur_val, slope, hours_left = predict_rul_linear(df, 'heat_exchanger_delta_t', threshold=5.0, mode='decreasing')
                if cur_val is not None:
                    st.metric("Delta Térmico Actual", f"{cur_val:.1f} °C", delta=f"{slope:.4f} °C / hora", delta_color="normal")
                    if hours_left == float('inf'):
                        st.success("✅ Intercambiador estable: Sin pérdida de transferencia térmica.")
                    else:
                        st.warning(f"⚠️ Limpieza de incrustaciones recomendada en aproximadamente **{hours_left:.1f} horas** (Umbral crítico: 5.0 °C).")
                        
                        df_proj = df[['timestamp', 'heat_exchanger_delta_t']].copy()
                        df_proj['timestamp'] = pd.to_datetime(df_proj['timestamp'])
                        last_time = df_proj['timestamp'].max()
                        future_times = [last_time + pd.Timedelta(hours=h) for h in np.linspace(0, max(1.0, hours_left), 50)]
                        future_vals = [cur_val + (slope * h) for h in np.linspace(0, max(1.0, hours_left), 50)]
                        df_future = pd.DataFrame({'timestamp': future_times, 'heat_exchanger_delta_t': future_vals, 'Tipo': 'Proyección (IA)'})
                        df_proj['Tipo'] = 'Histórico'
                        df_comb = pd.concat([df_proj, df_future])
                        
                        fig_proj = px.line(df_comb, x='timestamp', y='heat_exchanger_delta_t', color='Tipo', 
                                           title='Proyección de Delta Térmico (Incrustación de Tubos)',
                                           color_discrete_map={'Histórico': '#1f77b4', 'Proyección (IA)': '#ef4444'})
                        fig_proj.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="Límite Mantenimiento (5 °C)")
                        st.plotly_chart(fig_proj, use_container_width=True)
            else:
                st.info("No hay datos de delta térmico del intercambiador disponibles.")
                
    with tab4:
        st.subheader("Agrupamiento de Estados Operacionales (K-Means)")
        st.markdown("El modelo K-Means agrupa los datos históricos para identificar automáticamente los perfiles de carga del motor.")
        
        cluster_features = st.multiselect("Sensores para clustering:", 
                                         options=[c for c in df.columns if c not in ['timestamp', 'is_anomaly', 'anomaly_score', 'cluster', 'latitude', 'longitude'] and not c.startswith('cylinder_')],
                                         default=['engine_rpm', 'engine_torque', 'fuel_rate'])
        n_clusters = st.slider("Número de clústeres:", min_value=2, max_value=5, value=3)
        
        if st.button("Ejecutar K-Means"):
            with st.spinner("Ejecutando K-Means..."):
                df_clustered = cluster_operating_states(df, cluster_features, n_clusters=n_clusters)
                if 'cluster' in df_clustered.columns:
                    df_clustered['cluster'] = df_clustered['cluster'].astype(str)
                    
                    # Graficar variables clave
                    col_x = cluster_features[0] if len(cluster_features) > 0 else 'engine_rpm'
                    col_y = cluster_features[1] if len(cluster_features) > 1 else 'engine_torque'
                    
                    fig_cluster = px.scatter(df_clustered, x=col_x, y=col_y, color='cluster',
                                             title=f"Perfiles Operacionales Clúster: {col_y} vs {col_x}",
                                             color_discrete_sequence=px.colors.qualitative.Safe)
                    st.plotly_chart(fig_cluster, use_container_width=True)
                    
                    # Mostrar distribución de registros por clúster con inferencia de etiqueta
                    counts = df_clustered['cluster'].value_counts()
                    st.markdown("#### Distribución de Tiempo por Clúster:")
                    cols = st.columns(len(counts))
                    for idx, (clust, count) in enumerate(counts.items()):
                        cluster_df = df_clustered[df_clustered['cluster'] == clust]
                        avg_rpm = cluster_df['engine_rpm'].mean() if 'engine_rpm' in cluster_df.columns else 0.0
                        avg_torque = cluster_df['engine_torque'].mean() if 'engine_torque' in cluster_df.columns else 0.0
                        
                        if avg_rpm < 500:
                            label = "💤 Ralentí / Parada"
                        elif avg_torque > 5000:
                            label = "🏋️ Alta Carga / Remolque"
                        else:
                            label = "🟢 Crucero ECO"
                            
                        with cols[idx]:
                            st.metric(f"Clúster {clust} ({label})", f"{count} min", f"{count/len(df_clustered)*100:.1f}%")
                            
    with tab5:
        st.subheader("Análisis de Balance Térmico de Cilindros (V16)")
        st.markdown("Monitorea la temperatura de los gases de escape individuales de los 16 cilindros en V para detectar desbalances en la inyección de combustible.")
        
        cyl_cols = [f"cylinder_{i}" for i in range(1, 17)]
        if any(c in df.columns for c in cyl_cols):
            # Obtener el último registro
            last_record = df.iloc[-1]
            temps = {f"Cil {i}": last_record[f"cylinder_{i}"] for i in range(1, 17) if f"cylinder_{i}" in df.columns}
            
            df_temps = pd.DataFrame(list(temps.items()), columns=["Cilindro", "Temperatura (°C)"])
            
            max_t = df_temps["Temperatura (°C)"].max()
            min_t = df_temps["Temperatura (°C)"].min()
            delta_t = max_t - min_t
            avg_t = df_temps["Temperatura (°C)"].mean()
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Temperatura Promedio", f"{avg_t:.1f} °C")
            with col_b2:
                st.metric("Temperatura Máxima", f"{max_t:.1f} °C")
            with col_b3:
                color_type = "inverse" if delta_t > 50.0 else "normal"
                st.metric("Desbalance Térmico Máximo", f"{delta_t:.1f} °C", 
                          delta="🚨 CRÍTICO (>50°C)" if delta_t > 50.0 else "🟢 NOMINAL", 
                          delta_color=color_type)
            
            if delta_t > 50.0:
                hottest_cil = df_temps.loc[df_temps["Temperatura (°C)"].idxmax()]["Cilindro"]
                st.error(f"🚨 **Alerta de Desbalance:** Se ha detectado una diferencia térmica de escape mayor a 50 °C. El cilindro más caliente es el **{hottest_cil} ({max_t:.1f} °C)**. Se requiere inspección del inyector y calibración de válvulas.")
            else:
                st.success("✅ **Balance Nominal:** La distribución de temperaturas de escape entre todos los cilindros está dentro de los límites de tolerancia normales.")
            
            # Graficar barras de cilindros
            fig_cyls = px.bar(df_temps, x="Cilindro", y="Temperatura (°C)", 
                              title="Temperatura de Escape por Cilindro (Último Registro)",
                              color="Temperatura (°C)", color_continuous_scale="Jet",
                              range_color=[100, 700])
            st.plotly_chart(fig_cyls, use_container_width=True)
            
            # Línea de tiempo de delta histórico
            df_historical_delta = df.copy()
            present_cyl_cols = [c for c in cyl_cols if c in df.columns]
            df_historical_delta['delta_cylinders'] = df_historical_delta[present_cyl_cols].max(axis=1) - df_historical_delta[present_cyl_cols].min(axis=1)
            
            fig_delta_hist = px.line(df_historical_delta, x="timestamp", y="delta_cylinders", 
                                     title="Historial del Desbalance Térmico Máximo",
                                     labels={"delta_cylinders": "Delta T (°C)"})
            fig_delta_hist.add_hline(y=50.0, line_dash="dash", line_color="red", annotation_text="Límite Tolerancia (50°C)")
            st.plotly_chart(fig_delta_hist, use_container_width=True)
        else:
            st.warning("⚠️ No se encontraron columnas de cilindros en los datos de InfluxDB.")
            
    with tab6:
        st.subheader("Indicadores Ecológicos y de Emisiones (SCR & ECO-Advisor)")
        st.markdown("Evalúa la eficiencia energética del motor CAT 3516B y el rendimiento del catalizador SCR en la mitigación de gases contaminantes (NOx).")
        
        col_e1, col_e2, col_e3 = st.columns(3)
        
        # 1. Nivel de Urea (DEF)
        def_level = df['def_level_percent'].iloc[-1] if 'def_level_percent' in df.columns else 0.0
        with col_e1:
            st.metric("Nivel de Urea (DEF)", f"{def_level:.1f} %")
            if def_level < 15.0:
                st.error("🚨 **Tanque de Urea Crítico:** Rellenar aditivo SCR inmediatamente.")
            else:
                st.success("🟢 Nivel de urea suficiente.")
                
        # 2. Reducción de NOx
        nox_raw = df['nox_raw_ppm'].mean() if 'nox_raw_ppm' in df.columns else 0.0
        nox_red = df['nox_reduced_ppm'].mean() if 'nox_reduced_ppm' in df.columns else 0.0
        prevented_nox = nox_raw - nox_red
        red_pct = (prevented_nox / nox_raw * 100) if nox_raw > 0 else 0.0
        
        with col_e2:
            st.metric("Reducción de NOx SCR", f"{red_pct:.1f} %", delta=f"-{prevented_nox:.1f} ppm")
            
        # 3. Consumo Específico Promedio
        avg_specific = df['specific_consumption'].mean() if 'specific_consumption' in df.columns else 0.0
        with col_e3:
            st.metric("Consumo Específico de Combustible", f"{avg_specific:.3f} L/kWh" if avg_specific > 0 else "N/A")
            
        # Graficar comparación NOx
        if 'nox_raw_ppm' in df.columns:
            df_nox_melt = df.melt(id_vars=['timestamp'], value_vars=['nox_raw_ppm', 'nox_reduced_ppm'],
                                  var_name='Tipo de NOx', value_name='Concentración (ppm)')
            df_nox_melt['Tipo de NOx'] = df_nox_melt['Tipo de NOx'].map({
                'nox_raw_ppm': 'NOx Crudo (Sin Tratar)',
                'nox_reduced_ppm': 'NOx Emitido (Post-SCR)'
            })
            
            fig_nox = px.line(df_nox_melt, x='timestamp', y='Concentración (ppm)', color='Tipo de NOx',
                               title='Rendimiento del Catalizador SCR (NOx Crudo vs. NOx Emitido)',
                               color_discrete_map={'NOx Crudo (Sin Tratar)': '#ef4444', 'NOx Emitido (Post-SCR)': '#10b981'})
            st.plotly_chart(fig_nox, use_container_width=True)
            
            # Gráfico de Urea
            if 'def_level_percent' in df.columns:
                fig_def = px.area(df, x='timestamp', y='def_level_percent', title='Nivel del Tanque de Urea (DEF)',
                                  labels={'def_level_percent': 'Urea (%)'}, color_discrete_sequence=['#3b82f6'])
                fig_def.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_def, use_container_width=True)
                
    with tab7:
        st.subheader("Geolocalización y Mapa de Estrés Operacional")
        st.markdown("Mapeo de la ruta del barco y distribución de variables de estrés mecánico a lo largo de la navegación.")
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df_gps = df.dropna(subset=['latitude', 'longitude']).copy()
            df_gps = df_gps[(df_gps['latitude'] != 0.0) & (df_gps['longitude'] != 0.0)]
            
            if not df_gps.empty:
                stress_var = st.selectbox("Selecciona la variable para mapear el estrés:", 
                                         options=['fuel_rate', 'vibration', 'engine_rpm', 'load_percent', 'roll', 'pitch'],
                                         index=0)
                
                # Graficar mapa Mapbox
                fig_map = px.scatter_mapbox(df_gps, 
                                            lat="latitude", 
                                            lon="longitude", 
                                            color=stress_var,
                                            size="sog" if "sog" in df_gps.columns else None,
                                            color_continuous_scale=px.colors.sequential.Jet,
                                            size_max=15, 
                                            zoom=11,
                                            title=f"Ruta del Remolcador ASD - Coloreada por {stress_var.upper()}",
                                            mapbox_style="open-street-map")
                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("⚠️ Los registros de GPS contienen coordenadas vacías o en cero (0.0).")
        else:
            st.warning("⚠️ No se encontraron columnas de GPS (latitude, longitude) en los datos de InfluxDB.")
            
    with tab8:
        st.subheader("📄 Informe Técnico Consolidado & Manual de Operación")
        st.markdown("Consolidación escrita automática de hallazgos del sistema de Inteligencia Artificial y guía de referencia rápida.")
        
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            st.markdown("### 📝 Diagnóstico de IA")
            start_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M') if 'timestamp' in df.columns else 'N/A'
            end_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M') if 'timestamp' in df.columns else 'N/A'
            total_records = len(df)
            anomalies_count = st.session_state.get('anomalies_count', 0)
            imp_df = st.session_state.get('importance_df', None)
            
            informe_md = f"""
*   **Período Analizado:** {start_date} al {end_date}
*   **Total de Registros Evaluados:** {total_records}
*   **Anomalías Severas Detectadas:** {anomalies_count}
"""
            if anomalies_count > 0: 
                informe_md += "*   🚨 **ALERTA DE INTEGRIDAD:** Se requiere inspección mecánica de los eventos anómalos reportados.\n"
                
            # Agregar V16 cylinder delta a informe
            cyl_cols = [f"cylinder_{i}" for i in range(1, 17)]
            if any(c in df.columns for c in cyl_cols):
                last_record = df.iloc[-1]
                cyl_temps = [last_record[c] for c in cyl_cols if c in df.columns]
                delta_cyl = max(cyl_temps) - min(cyl_temps)
                informe_md += f"*   **Desbalance Térmico Cilindros V16:** {delta_cyl:.1f} °C\n"
                if delta_cyl > 50.0:
                    informe_md += "    *   🚨 **ATENCIÓN:** Exceso de desbalance térmico detectado en inyección de cilindros.\n"
                    
            # Agregar RUL del filtro
            if 'filter_clog' in df.columns:
                cur_val, slope, hours_left = predict_rul_linear(df, 'filter_clog', threshold=90.0, mode='increasing')
                if cur_val is not None and hours_left != float('inf'):
                    informe_md += f"*   🔮 **Predicción RUL Filtro Aceite:** Relleno de filtro proyectado en {hours_left:.1f} horas.\n"
            
            informe_md += "\n#### Causa Raíz Predominante (Random Forest)\n"
            if imp_df is not None and not imp_df.empty:
                for i, row in imp_df.head(3).iterrows():
                    sensor_clean = SENSOR_NAMES.get(row['Sensor'], row['Sensor'])
                    informe_md += f"*   **{sensor_clean}**: {row['Importancia']*100:.1f}% de impacto\n"
            else:
                informe_md += "*   *Ejecuta el análisis de importancia en la pestaña Predictivo para calcular.* \n"
                
            st.info(informe_md)
            
            # Mostrar Historial de Mantenimiento si existe
            df_maint = st.session_state.get('df_maint', pd.DataFrame())
            if not df_maint.empty:
                st.markdown("#### 🛠️ Historial de Mantenimiento Registrado (InfluxDB)")
                df_maint_show = df_maint.copy()
                if 'timestamp' in df_maint_show.columns:
                    try:
                        df_maint_show['timestamp'] = pd.to_datetime(df_maint_show['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                cols = [c for c in ['timestamp', 'component', 'event_name'] if c in df_maint_show.columns]
                st.dataframe(df_maint_show[cols], use_container_width=True)
                
            # Botón de Descarga PDF
            try:
                from report_generator import generate_pdf
                pdf_bytes = generate_pdf(df, anomalies_count, imp_df)
                st.download_button(
                    label="📥 Descargar Informe Técnico Completo (PDF)",
                    data=pdf_bytes,
                    file_name=f"Reporte_Tecnico_Avanzado_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error al habilitar descarga de reporte PDF: {e}")
                
        with col_i2:
            st.markdown("### 📖 Guía de Referencia Rápida")
            st.markdown("""
El **AI Marine Analyzer** evalúa múltiples variables simultáneamente para predecir fallas antes de que ocurran.

#### Flujo Operativo Rápido:
1.  **Definir Ventana Temporal:** Elige el número de días en el panel izquierdo.
2.  **Cargar Datos:** Haz clic en "Cargar Datos de InfluxDB".
3.  **Analizar:** Explora anomalías, balance de cilindros V16, mapa de ruta y RUL.

#### Diagnósticos Comunes:
| Síntoma | Algoritmo / Vista | Causa Probable | Acción Recomendada |
| :--- | :--- | :--- | :--- |
| Delta Cilindros > 50°C | Balance V16 | Falla inyector / desbalance | Test de compresión o desmontar inyector |
| RUL Filtro < 24 hrs | Predictivo & RUL | Saturación física del filtro | Reemplazar elemento filtrante de aceite |
| Pérdida Delta Intercambiador | Predictivo & RUL | Incrustación de intercambiador | Limpieza del haz de tubos (agua mar) |
| NOx Crudo = NOx Emitido | ECO & SCR | Tanque Urea vacío o falla SCR | Rellenar Urea (DEF) o revisar dosificador |
""")
            try:
                from report_generator import generate_manual_pdf
                manual_bytes = generate_manual_pdf()
                st.download_button(
                    label="📥 Descargar Manual de Operación (PDF)",
                    data=manual_bytes,
                    file_name="Manual_de_Operacion_AI_Marine_Analyzer.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error al habilitar descarga de manual: {e}")

else:
    st.info("👈 Utiliza el panel izquierdo para descargar los datos desde InfluxDB y comenzar el análisis.")
