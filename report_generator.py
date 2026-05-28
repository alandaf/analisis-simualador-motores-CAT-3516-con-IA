from fpdf import FPDF
from datetime import datetime
import pandas as pd

SENSOR_NAMES = {
    'oil_pressure': 'Presión de Lubricación',
    'engine_rpm': 'RPM del Motor',
    'oil_filter_clogging': 'Taponamiento del Filtro de Aceite',
    'exhaust_temperature': 'Temperatura de Escape',
    'vibration': 'Nivel de Vibración del Motor',
    'engine_temperature': 'Temperatura del Refrigerante',
    'load_percent': 'Carga del Motor',
    'battery_voltage': 'Voltaje del Sistema',
    'fuel_rate': 'Tasa de Flujo de Combustible',
    'intake_pressure': 'Presión de Admisión (Turbo)',
    'intake_temperature': 'Temperatura de Admisión',
    'engine_torque': 'Torque del Eje',
    'exhaust_pressure': 'Presión de Gases de Escape',
    'fuel_efficiency': 'Eficiencia Térmica del Combustible'
}

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(20, 50, 100)
        self.cell(0, 10, 'INFORME TÉCNICO DE INTELIGENCIA ARTIFICIAL', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Sistema de Mantenimiento Predictivo - CAT 3516B', 0, 1, 'C')
        self.line(10, 28, 200, 28)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Página {self.page_no()} | Generado automáticamente por MotorData AI', 0, 0, 'C')

def generate_pdf(df, anomalies_count, importance_df):
    pdf = PDFReport()
    pdf.add_page()
    
    # Cálculos dinámicos
    start_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M') if 'timestamp' in df.columns else 'N/A'
    end_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M') if 'timestamp' in df.columns else 'N/A'
    total_records = len(df)
    
    max_rpm = df['engine_rpm'].max() if 'engine_rpm' in df.columns else 0
    avg_temp = df['engine_temperature'].mean() if 'engine_temperature' in df.columns else 0
    avg_oil = df['oil_pressure'].mean() if 'oil_pressure' in df.columns else 0
    
    # 1. Resumen Operacional y Estadísticas
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, '1. RESUMEN OPERACIONAL Y ESTADÍSTICO', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, f"El presente documento consolida el análisis de {total_records} registros de telemetría extraídos "
                         f"desde la base de datos InfluxDB, abarcando el período entre {start_date} y {end_date}. "
                         f"Los datos fueron agrupados y procesados utilizando algoritmos de Machine Learning "
                         f"para identificar patrones ocultos de desgaste mecánico.")
    pdf.ln(2)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 6, f"  - RPM Máximas Registradas: {max_rpm:.1f} RPM", 0, 1)
    pdf.cell(0, 6, f"  - Temperatura Promedio del Sistema: {avg_temp:.1f} C", 0, 1)
    pdf.cell(0, 6, f"  - Presión de Aceite Promedio: {avg_oil:.1f} PSI", 0, 1)
    pdf.ln(6)
    
    # 2. Isolation Forest
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, '2. DIAGNÓSTICO DE INTEGRIDAD (ISOLATION FOREST)', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, "Se ha ejecutado el algoritmo no supervisado 'Isolation Forest' para mapear la telemetría en "
                         "múltiples dimensiones. Este modelo detecta cuando el motor se comporta de una manera que "
                         "estadísticamente rompe su patrón normal de operación conjunta.")
    
    pdf.set_font('Arial', 'B', 10)
    if anomalies_count > 0:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, f"RESULTADO: Se han detectado {anomalies_count} eventos críticos anómalos.", 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, "Estos eventos representan picos de inestabilidad donde las variables físicas (como presión, "
                             "temperatura o vibración) presentaron divergencias severas. Es imperativo revisar la "
                             "bitácora de mantenimiento para estos periodos de tiempo específicos.")
    else:
        pdf.set_text_color(0, 120, 0)
        pdf.cell(0, 8, f"RESULTADO: No se detectaron anomalías severas ({anomalies_count} eventos).", 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, "El motor ha operado dentro de los límites de tolerancia estadística estipulados "
                             "por el modelo de aprendizaje automático.")
    pdf.ln(6)
    
    # 3. Random Forest (Root Cause)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, '3. ANÁLISIS DE CAUSA RAÍZ (RANDOM FOREST)', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, "Mediante un ensamble de árboles de decisión (Random Forest Regressor), la IA ha evaluado "
                         "el impacto no-lineal de todos los sensores sobre las variables críticas del motor para "
                         "determinar matemáticamente el origen de las fallas (Feature Importance).")
    
    if importance_df is not None and not importance_df.empty:
        top_sensor = importance_df.iloc[0]['Sensor']
        top_impact = importance_df.iloc[0]['Importancia'] * 100
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, 'Top 3 Parametros Físicos con Mayor Impacto:', 0, 1)
        pdf.set_font('Arial', '', 10)
        for i, row in importance_df.head(3).iterrows():
            sensor_clean = SENSOR_NAMES.get(row['Sensor'], row['Sensor'])
            pdf.cell(0, 6, f"  - Parametro: {sensor_clean} ({row['Importancia']*100:.1f}% de correlación)", 0, 1)
            
        pdf.ln(4)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'CONCLUSIÓN TÉCNICA Y RECOMENDACIÓN:', 0, 1)
        pdf.set_font('Arial', '', 10)
        top_sensor_clean = SENSOR_NAMES.get(top_sensor, top_sensor).lower()
        pdf.multi_cell(0, 6, f"El algoritmo ha determinado con alta confianza que una fluctuacion en la '{top_sensor_clean}' "
                             f"es el agente causal primario responsable del {top_impact:.1f}% del comportamiento "
                             f"anormal estudiado. Se recomienda dirigir los esfuerzos de inspección mecánica y mantenimiento preventivo "
                             f"inmediatamente hacia los subsistemas asociados a la {top_sensor_clean}.")
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 8, 'Nota: No se ejecutó el análisis de Mantenimiento Predictivo antes de generar este reporte.', 0, 1)
        
    return pdf.output(dest='S').encode('latin1')
