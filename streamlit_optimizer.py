"""
OPTIMIZADOR DE ESTRATEGIA DE MUESTREO
Aplicación Streamlit MILP Interactiva
Basada en especificación del modelo Word
"""

import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import plotly.graph_objects as go
import plotly.express as px

# Configuración de página
st.set_page_config(
    page_title="Optimizador de Muestreo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1em;
        color: #1f77b4;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="main-header">🎯 Optimizador de Estrategia de Muestreo</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SECCIÓN 1: PARÁMETROS FIJOS (DEL DOCUMENTO)
# ============================================================================

st.header("📊 1. Parámetros de Entrada (Valores Fijos)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⏱️ Parámetros de Tiempo")
    
    T_total_days = 60  # Días
    T_total = T_total_days * 8  # Convertir a horas
    h_work = 8  # Horas efectivas/día
    t_standard = 1  # Horas por punto estándar
    t_multilevel = 3  # Horas por punto multinivel
    t_24h = 25  # Horas por medición 24h
    t_extended = 121  # Horas por medición extendida
    
    st.metric("Tiempo total disponible", f"{T_total_days} días ({T_total} horas)")
    st.metric("Horas de trabajo por día", f"{h_work} horas")
    st.metric("Tiempo - Punto estándar", f"{t_standard} hora")
    st.metric("Tiempo - Punto multinivel", f"{t_multilevel} horas")
    st.metric("Tiempo - Medición 24h", f"{t_24h} horas")
    st.metric("Tiempo - Medición extendida", f"{t_extended} horas")

with col2:
    st.subheader("💰 Parámetros de Presupuesto")
    
    B_total = 400_000_000  # COP
    daily_cost = 1_350_000  # COP
    cost_chromatography = 1_069_200
    cost_deuterium = 792_000
    cost_helium = 3_240_000
    cost_mineralogy = 309_200
    cost_biogeochemistry = 3_100_000
    
    costs_lab = {
        'Cromatografía': cost_chromatography,
        'Deuterio': cost_deuterium,
        'Helio': cost_helium,
        'Mineralogía': cost_mineralogy,
        'Biogeoquímica': cost_biogeochemistry
    }
    
    st.metric("Presupuesto total", f"${B_total:,.0f} COP")
    st.metric("Costo operativo diario", f"${daily_cost:,.0f} COP")
    
    with st.expander("Ver costos de laboratorio", expanded=False):
        for lab, cost in costs_lab.items():
            st.write(f"• {lab}: ${cost:,}")

# ============================================================================
# DATOS DE ENTRADA
# ============================================================================

st.markdown("---")
st.header("🔧 2. Datos de Entrada")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 Círculos de Hadas")
    
    # Opción: Usar datos predefinidos o cargar
    circle_option = st.radio("Selecciona opción:", ["Usar datos predefinidos", "Cargar CSV"])
    
    if circle_option == "Usar datos predefinidos":
        circles_data = {
            'C1': 113, 'C2': 164, 'C3': 702, 'C4': 1393, 'C5': 221,
            'C6': 310, 'C7': 204, 'C8': 227, 'C9': 110, 'C10': 314,
            'C11': 206, 'C12': 113, 'C13': 437, 'C14': 156, 'C15': 274,
            'C16': 135, 'C17': 106, 'C18': 356, 'C19': 696, 'C20': 165,
            'C21': 346, 'C22': 199, 'C23': 162, 'C24': 204, 'C25': 48.1,
            'C26': 308, 'C27': 364, 'C28': 101, 'C29': 163, 'C30': 63
        }
        st.success(f"✓ {len(circles_data)} círculos cargados")
    else:
        uploaded_file = st.file_uploader("Sube CSV con círculos", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            circles_data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        else:
            circles_data = {}
    
    # Mostrar preview
    if circles_data:
        df_circles = pd.DataFrame(list(circles_data.items()), columns=['Círculo', 'Perímetro (m)'])
        st.dataframe(df_circles.head(10), use_container_width=True)
        st.caption(f"Total: {len(circles_data)} círculos")

with col2:
    st.subheader("🧪 Tipos de Análisis")
    
    lab_types = list(costs_lab.keys())
    
    # Checkbox para seleccionar análisis
    selected_labs = st.multiselect(
        "Selecciona análisis a incluir:",
        lab_types,
        default=lab_types
    )
    
    st.success(f"✓ {len(selected_labs)} análisis seleccionados")

with col3:
    st.subheader("📏 Separaciones Permitidas")
    
    separations_default = [100, 150, 200, 300, 500]
    
    # Opción para modificar separaciones
    sep_text = st.text_input(
        "Separaciones (en metros, separadas por comas):",
        value=", ".join(map(str, separations_default))
    )
    
    try:
        separations = [int(x.strip()) for x in sep_text.split(",")]
        st.success(f"✓ {len(separations)} separaciones: {separations} m")
    except:
        st.error("❌ Error en formato de separaciones")
        separations = separations_default

# ============================================================================
# MÍNIMOS REQUERIDOS
# ============================================================================

st.markdown("---")
st.header("⚙️ 3. Mínimos Requeridos")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Muestreo de Campo")
    min_multilevel_per_circle = st.number_input(
        "Puntos multinivel mín. por círculo:",
        value=3, min_value=1, max_value=10
    )
    min_24h_total = st.number_input(
        "Mediciones 24h mínimas (total):",
        value=10, min_value=1, max_value=50
    )
    min_extended_total = st.number_input(
        "Mediciones extendidas mín. (total):",
        value=2, min_value=0, max_value=20
    )

with col2:
    st.subheader("Laboratorio")
    min_samples = {}
    for lab in selected_labs:
        if lab == 'Cromatografía':
            default_val = 15
        elif lab == 'Deuterio':
            default_val = 5
        elif lab == 'Helio':
            default_val = 5
        elif lab == 'Mineralogía':
            default_val = 15
        else:  # Biogeoquímica
            default_val = 15
        
        min_samples[lab] = st.number_input(
            f"Muestras mín. {lab}:",
            value=default_val, min_value=1, max_value=100
        )

with col3:
    st.subheader("Recursos")
    max_equipment = st.number_input(
        "Equipos máximos disponibles:",
        value=4, min_value=1, max_value=10
    )
    st.caption("(2 personas por equipo)")

# ============================================================================
# PESOS DE CALIDAD (AHP)
# ============================================================================

st.markdown("---")
st.header("📈 4. Pesos de Calidad Científica (AHP)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Muestreo de Campo")
    w_standard = st.slider("Peso - Punto estándar:", 0.0, 1.0, 0.2417, 0.0001)
    w_multilevel = st.slider("Peso - Punto multinivel:", 0.0, 1.0, 0.2029, 0.0001)
    w_24h = st.slider("Peso - Medición 24h:", 0.0, 1.0, 0.1459, 0.0001)
    w_extended = st.slider("Peso - Medición extendida:", 0.0, 1.0, 0.0523, 0.0001)

with col2:
    st.subheader("Análisis de Laboratorio")
    w_lab = {}
    for lab in selected_labs:
        default_weights = {
            'Cromatografía': 0.1511,
            'Deuterio': 0.0897,
            'Helio': 0.0438,
            'Mineralogía': 0.0399,
            'Biogeoquímica': 0.0328
        }
        w_lab[lab] = st.slider(
            f"Peso - {lab}:",
            0.0, 1.0, default_weights.get(lab, 0.05), 0.0001
        )

# Mostrar suma de pesos
total_weight = w_standard + w_multilevel + w_24h + w_extended + sum(w_lab.values())
st.info(f"📊 Suma total de pesos: {total_weight:.4f} (para referencia)")

# ============================================================================
# BOTÓN PARA EJECUTAR OPTIMIZACIÓN
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.subheader("🚀 Ejecutar Optimización")
    if st.button("▶ Resolver Modelo MILP", key="solve_button", use_container_width=True):
        
        # Validaciones
        if not circles_data:
            st.error("❌ Debes cargar círculos de hadas")
            st.stop()
        
        if not selected_labs:
            st.error("❌ Debes seleccionar al menos un análisis")
            st.stop()
        
        circles = list(circles_data.keys())
        
        # ============================================================
        # CONSTRUCCIÓN DEL MODELO MILP
        # ============================================================
        
        with st.spinner("⏳ Construyendo modelo..."):
            
            # Función auxiliar
            def calculate_standard_points(perimeter, separation):
                n_points = max(1, int(np.floor(perimeter / separation)))
                return n_points
            
            # Matriz de puntos estándar
            standard_points = {}
            for circle in circles:
                standard_points[circle] = {}
                perimeter = circles_data[circle]
                for sep in separations:
                    standard_points[circle][sep] = calculate_standard_points(perimeter, sep)
            
            # Crear modelo
            model = LpProblem("Sampling_Optimization", LpMaximize)
            
            # Variables de decisión
            x = LpVariable.dicts("separation_choice",
                                ((circle, sep) for circle in circles for sep in separations),
                                cat='Binary')
            y = LpVariable.dicts("multilevel_points", circles, lowBound=0, cat='Integer')
            E = LpVariable("num_equipment", lowBound=1, upBound=max_equipment, cat='Integer')
            W = LpVariable("total_work_hours", lowBound=0, cat='Integer')
            M24 = LpVariable("total_24h_measurements", lowBound=0, cat='Integer')
            ME = LpVariable("total_extended_measurements", lowBound=0, cat='Integer')
            S = LpVariable.dicts("lab_samples", selected_labs, lowBound=0, cat='Integer')
            
            # Función objetivo
            objective = (
                w_standard * lpSum([
                    standard_points[circle][sep] * x[circle, sep]
                    for circle in circles for sep in separations
                ]) +
                w_multilevel * lpSum([y[circle] for circle in circles]) +
                w_24h * M24 +
                w_extended * ME +
                sum([w_lab[lab] * S[lab] for lab in selected_labs])
            )
            
            model += objective, "Total_Scientific_Quality"
            
            # Restricciones
            # R1: Definición de esfuerzo
            model += (
                W == lpSum([
                    standard_points[circle][sep] * t_standard * x[circle, sep]
                    for circle in circles for sep in separations
                ]) +
                lpSum([y[circle] * t_multilevel for circle in circles]) +
                M24 * t_24h +
                ME * t_extended,
                "R1_Total_Hours"
            )
            
            # R2: Capacidad operativa
            model += W <= E * h_work * T_total, "R2_Operational_Capacity"
            
            # R3: Presupuesto
            filtered_costs = {lab: costs_lab[lab] for lab in selected_labs}
            model += (
                daily_cost * (W / h_work) +
                sum([filtered_costs[lab] * S[lab] for lab in selected_labs])
                <= B_total,
                "R3_Budget"
            )
            
            # R4: Selección única de separación
            for circle in circles:
                model += lpSum([x[circle, sep] for sep in separations]) == 1, f"R4_Unique_Sep_{circle}"
            
            # R5: Mínimos de mediciones
            model += M24 >= min_24h_total, "R5_Min_24h"
            model += ME >= min_extended_total, "R5_Min_Extended"
            
            # R6: Mínimos de laboratorio
            for lab in selected_labs:
                model += S[lab] >= min_samples[lab], f"R6_Min_Lab_{lab}"
            
            # R7: Mínimo multinivel por círculo
            for circle in circles:
                model += y[circle] >= min_multilevel_per_circle, f"R7_Min_Multilevel_{circle}"
            
            # R8: Máximo de equipos
            model += E <= max_equipment, "R8_Max_Equipment"
        
        # ============================================================
        # RESOLVER MODELO
        # ============================================================
        
        with st.spinner("🔄 Resolviendo... (esto puede tomar 30-60 segundos)"):
            model.solve(PULP_CBC_CMD(msg=0))
        
        # ============================================================
        # MOSTRAR RESULTADOS
        # ============================================================
        
        if model.status == 1:
            st.markdown('<div class="success-box"><h3>✅ ¡SOLUCIÓN ÓPTIMA ENCONTRADA!</h3></div>', unsafe_allow_html=True)
            
            # ============================================================
            # TAB 1: RESUMEN EJECUTIVO
            # ============================================================
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Resumen", "🎯 Estrategia", "💰 Costos", "📈 Gráficos", "📥 Descargas"
            ])
            
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Calidad Científica",
                        f"{value(model.objective):.4f}",
                        "Maximizado"
                    )
                
                with col2:
                    st.metric(
                        "Equipos Necesarios",
                        f"{int(E.varValue)}",
                        f"{int(E.varValue)*2} personas"
                    )
                
                with col3:
                    work_hours = int(W.varValue)
                    work_days = work_hours / h_work
                    st.metric(
                        "Horas Totales",
                        f"{work_hours}",
                        f"{work_days:.1f} días"
                    )
                
                with col4:
                    operational_cost = daily_cost * (work_hours / h_work)
                    lab_cost = sum([filtered_costs[lab] * int(S[lab].varValue) for lab in selected_labs])
                    total_cost = operational_cost + lab_cost
                    st.metric(
                        "Costo Total",
                        f"${total_cost:,.0f}",
                        f"{(total_cost/B_total)*100:.1f}% presupuesto"
                    )
                
                # Detalles adicionales
                st.subheader("Mediciones de Campo")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mediciones 24h", int(M24.varValue))
                with col2:
                    st.metric("Mediciones Extendidas", int(ME.varValue))
                with col3:
                    total_points = sum([
                        standard_points[circle][sep] * x[circle, sep].varValue
                        for circle in circles for sep in separations
                    ]) + sum([int(y[circle].varValue) for circle in circles])
                    st.metric("Total Puntos", int(total_points))
                
                st.subheader("Muestras de Laboratorio")
                lab_summary = []
                for lab in selected_labs:
                    samples = int(S[lab].varValue)
                    cost = filtered_costs[lab] * samples
                    lab_summary.append({
                        'Análisis': lab,
                        'Muestras': samples,
                        'Costo ($)': cost
                    })
                
                df_lab = pd.DataFrame(lab_summary)
                st.dataframe(df_lab, use_container_width=True)
            
            # ============================================================
            # TAB 2: ESTRATEGIA POR CÍRCULO
            # ============================================================
            
            with tab2:
                strategy_data = []
                for circle in circles:
                    for sep in separations:
                        if x[circle, sep].varValue == 1:
                            n_standard = standard_points[circle][sep]
                            n_multi = int(y[circle].varValue)
                            strategy_data.append({
                                'Círculo': circle,
                                'Separación (m)': sep,
                                'Puntos Estándar': n_standard,
                                'Puntos Multinivel': n_multi,
                                'Total Puntos': n_standard + n_multi,
                                'Perímetro (m)': circles_data[circle]
                            })
                
                df_strategy = pd.DataFrame(strategy_data)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(df_strategy, use_container_width=True)
                
                with col2:
                    st.metric("Total Círculos", len(df_strategy))
                    st.metric("Total Puntos", df_strategy['Total Puntos'].sum())
                    st.metric("Separación Promedio", f"{df_strategy['Separación (m)'].mean():.0f}m")
            
            # ============================================================
            # TAB 3: ANÁLISIS DE COSTOS
            # ============================================================
            
            with tab3:
                work_hours = int(W.varValue)
                work_days = work_hours / h_work
                operational_cost = daily_cost * work_days
                lab_cost = sum([filtered_costs[lab] * int(S[lab].varValue) for lab in selected_labs])
                total_cost = operational_cost + lab_cost
                margin = B_total - total_cost
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Desglose de Costos")
                    cost_breakdown = pd.DataFrame({
                        'Concepto': [
                            f'Operativo ({work_days:.0f} días)',
                            'Laboratorio',
                            'TOTAL',
                            'Presupuesto Disponible',
                            'Margen'
                        ],
                        'Valor ($)': [
                            operational_cost,
                            lab_cost,
                            total_cost,
                            B_total,
                            margin
                        ]
                    })
                    
                    st.dataframe(cost_breakdown, use_container_width=True)
                
                with col2:
                    st.subheader("Indicadores")
                    utilization_time = (work_hours / (int(E.varValue) * h_work * T_total)) * 100
                    utilization_budget = (total_cost / B_total) * 100
                    
                    st.metric("Utilización de Tiempo", f"{utilization_time:.1f}%")
                    st.metric("Utilización de Presupuesto", f"{utilization_budget:.1f}%")
                    
                    if utilization_budget > 90:
                        st.warning("⚠️ Presupuesto muy ajustado")
                    elif utilization_budget > 80:
                        st.info("ℹ️ Presupuesto en rango normal")
                    else:
                        st.success("✓ Presupuesto con margen")
            
            # ============================================================
            # TAB 4: GRÁFICOS
            # ============================================================
            
            with tab4:
                col1, col2 = st.columns(2)
                
                # Gráfico 1: Composición de costos
                with col1:
                    fig1 = go.Figure(data=[go.Pie(
                        labels=['Operativo', 'Laboratorio'],
                        values=[operational_cost, lab_cost],
                        hole=0.3,
                        marker=dict(colors=['#1f77b4', '#ff7f0e'])
                    )])
                    fig1.update_layout(
                        title="Composición de Costos",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Gráfico 2: Utilización de recursos
                with col2:
                    fig2 = go.Figure(data=[
                        go.Bar(x=['Tiempo', 'Presupuesto'],
                               y=[utilization_time, utilization_budget],
                               marker=dict(color=['#2ca02c', '#d62728']))
                    ])
                    fig2.update_layout(
                        title="Utilización de Recursos (%)",
                        yaxis=dict(range=[0, 100]),
                        height=400,
                        showlegend=False,
                        hovermode='x unified'
                    )
                    fig2.add_hline(y=80, line_dash="dash", line_color="orange", 
                                   annotation_text="Alerta (80%)")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Gráfico 3: Muestras de laboratorio
                if lab_summary:
                    fig3 = px.bar(
                        df_lab,
                        x='Análisis',
                        y='Muestras',
                        title='Muestras por Tipo de Análisis',
                        color='Análisis',
                        height=400
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Gráfico 4: Separaciones elegidas
                separations_count = df_strategy['Separación (m)'].value_counts().sort_index()
                fig4 = px.bar(
                    x=separations_count.index,
                    y=separations_count.values,
                    title='Distribución de Separaciones Elegidas',
                    labels={'x': 'Separación (m)', 'y': 'Cantidad de Círculos'},
                    height=400
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            # ============================================================
            # TAB 5: DESCARGAS
            # ============================================================
            
            with tab5:
                st.subheader("📥 Descargar Resultados")
                
                # Preparar datos para descargar
                excel_buffer = pd.ExcelWriter('resultados.xlsx', engine='openpyxl')
                
                df_strategy.to_excel(excel_buffer, sheet_name='Estrategia', index=False)
                df_lab.to_excel(excel_buffer, sheet_name='Laboratorio', index=False)
                cost_breakdown.to_excel(excel_buffer, sheet_name='Costos', index=False)
                
                excel_buffer.close()
                
                with open('resultados.xlsx', 'rb') as f:
                    st.download_button(
                        label="📊 Descargar Excel con Resultados",
                        data=f.read(),
                        file_name="resultados_optimizacion.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                # Resumen en texto
                st.subheader("📝 Resumen de Resultados")
                
                summary_text = f"""
                **OPTIMIZACIÓN COMPLETADA**
                
                **Función Objetivo:** {value(model.objective):.4f}
                
                **Configuración Operativa:**
                - Equipos: {int(E.varValue)} ({int(E.varValue)*2} personas)
                - Horas totales: {work_hours} horas
                - Días de operación: {work_days:.1f} días
                
                **Muestreo de Campo:**
                - Total de puntos: {int(total_points)}
                - Mediciones 24h: {int(M24.varValue)}
                - Mediciones extendidas: {int(ME.varValue)}
                
                **Análisis de Laboratorio:**
                - Análisis seleccionados: {len(selected_labs)}
                - Total de muestras: {sum([int(S[lab].varValue) for lab in selected_labs])}
                
                **Costos:**
                - Costo operativo: ${operational_cost:,.0f}
                - Costo laboratorio: ${lab_cost:,.0f}
                - Costo total: ${total_cost:,.0f}
                - Presupuesto disponible: ${B_total:,.0f}
                - Margen: ${margin:,.0f} ({(margin/B_total)*100:.1f}%)
                
                **Utilización:**
                - Tiempo: {utilization_time:.1f}%
                - Presupuesto: {utilization_budget:.1f}%
                """
                
                st.text(summary_text)
        
        else:
            st.error(f"❌ No se encontró solución óptima. Estado: {model.status}")
            st.info("💡 Intenta ajustar los parámetros (presupuesto, tiempo, mínimos requeridos)")

# ============================================================================
# SIDEBAR: INFORMACIÓN
# ============================================================================

with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ Información")
    
    st.markdown("""
    **Optimizador de Muestreo**
    
    Aplicación basada en:
    - Programación Lineal Entera Mixta (MILP)
    - Análisis Jerárquico de Procesos (AHP)
    - PuLP + Streamlit
    
    **Características:**
    - Optimización en tiempo real
    - Visualizaciones interactivas
    - Exportación a Excel
    - Análisis sensibilidad
    """)
    
    st.markdown("---")
    st.markdown("**Versión:** 1.0")
    st.markdown("**Última actualización:** 2024")
