import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Optimizador de Muestreo",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Optimizador de Estrategia de Muestreo")
st.markdown("---")

# FIXED PARAMETERS FROM WORD
st.header("📊 Parametros Fijos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tiempos")
    T_total_days = 60
    T_total = T_total_days * 8
    h_work = 8
    t_standard = 1
    t_multilevel = 3
    t_24h = 25
    t_extended = 121
    
    st.write(f"**Tiempo total:** {T_total_days} dias = {T_total} horas")
    st.write(f"**Horas/dia:** {h_work}")
    st.write(f"**Punto estandar:** {t_standard} hora")
    st.write(f"**Punto multinivel:** {t_multilevel} horas")
    st.write(f"**Medicion 24h:** {t_24h} horas")
    st.write(f"**Medicion extendida:** {t_extended} horas")

with col2:
    st.subheader("Presupuesto")
    B_total = 400_000_000
    daily_cost = 1_350_000
    
    costs_lab = {
        'Cromatografia': 1_069_200,
        'Deuterio': 792_000,
        'Helio': 3_240_000,
        'Mineralogia': 309_200,
        'Biogeoquimica': 3_100_000
    }
    
    st.write(f"**Total:** ${B_total:,} COP")
    st.write(f"**Diario:** ${daily_cost:,} COP")
    for lab, cost in costs_lab.items():
        st.write(f"**{lab}:** ${cost:,}")

# INPUT DATA
st.markdown("---")
st.header("Datos de Entrada")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Circulos")
    circles_data = {
        'C1': 113, 'C2': 164, 'C3': 702, 'C4': 1393, 'C5': 221,
        'C6': 310, 'C7': 204, 'C8': 227, 'C9': 110, 'C10': 314,
        'C11': 206, 'C12': 113, 'C13': 437, 'C14': 156, 'C15': 274,
        'C16': 135, 'C17': 106, 'C18': 356, 'C19': 696, 'C20': 165,
        'C21': 346, 'C22': 199, 'C23': 162, 'C24': 204, 'C25': 48.1,
        'C26': 308, 'C27': 364, 'C28': 101, 'C29': 163, 'C30': 63
    }
    st.success(f"{len(circles_data)} circulos cargados")

with col2:
    st.subheader("Analisis")
    lab_types = list(costs_lab.keys())
    selected_labs = st.multiselect("Selecciona:", lab_types, default=lab_types)

with col3:
    st.subheader("Separaciones")
    separations = [100, 150, 200, 300, 500]
    st.write(f"Opciones: {separations} metros")

# MINIMUMS
st.markdown("---")
st.header("Minimos Requeridos")

col1, col2, col3 = st.columns(3)

with col1:
    min_multilevel_per_circle = st.slider("Multinivel/circulo:", 1, 10, 3)
    min_24h_total = st.slider("Mediciones 24h:", 1, 50, 10)
    min_extended_total = st.slider("Mediciones extendidas:", 0, 20, 2)

with col2:
    st.subheader("Laboratorio")
    min_samples = {}
    for lab in selected_labs:
        default = 15 if lab in ['Cromatografia', 'Mineralogia', 'Biogeoquimica'] else 5
        min_samples[lab] = st.number_input(f"{lab}:", value=default, min_value=1)

with col3:
    max_equipment = st.slider("Equipos maximos:", 1, 10, 4)

# QUALITY WEIGHTS
st.markdown("---")
st.header("Pesos de Calidad (AHP)")

col1, col2 = st.columns(2)

with col1:
    w_standard = st.slider("Punto estandar:", 0.0, 1.0, 0.2417)
    w_multilevel = st.slider("Punto multinivel:", 0.0, 1.0, 0.2029)
    w_24h = st.slider("Medicion 24h:", 0.0, 1.0, 0.1459)
    w_extended = st.slider("Medicion extendida:", 0.0, 1.0, 0.0523)

with col2:
    w_lab = {
        'Cromatografia': st.slider("Cromatografia:", 0.0, 1.0, 0.1511),
        'Deuterio': st.slider("Deuterio:", 0.0, 1.0, 0.0897),
        'Helio': st.slider("Helio:", 0.0, 1.0, 0.0438),
        'Mineralogia': st.slider("Mineralogia:", 0.0, 1.0, 0.0399),
        'Biogeoquimica': st.slider("Biogeoquimica:", 0.0, 1.0, 0.0328),
    }

# SOLVE BUTTON
st.markdown("---")

if st.button("Resolver Modelo MILP", use_container_width=True):
    if not selected_labs:
        st.error("Selecciona al menos un analisis")
        st.stop()
    
    circles = list(circles_data.keys())
    
    with st.spinner("Resolviendo..."):
        try:
            def calc_points(perim, sep):
                return max(1, int(np.floor(perim / sep)))
            
            standard_points = {}
            for circle in circles:
                standard_points[circle] = {}
                for sep in separations:
                    standard_points[circle][sep] = calc_points(circles_data[circle], sep)
            
            model = LpProblem("Sampling", LpMaximize)
            
            x = LpVariable.dicts("x", ((c, s) for c in circles for s in separations), cat='Binary')
            y = LpVariable.dicts("y", circles, lowBound=0, cat='Integer')
            E = LpVariable("E", lowBound=1, upBound=max_equipment, cat='Integer')
            W = LpVariable("W", lowBound=0, cat='Integer')
            M24 = LpVariable("M24", lowBound=0, cat='Integer')
            ME = LpVariable("ME", lowBound=0, cat='Integer')
            S = LpVariable.dicts("S", selected_labs, lowBound=0, cat='Integer')
            
            obj = (
                w_standard * lpSum([standard_points[c][s] * x[c,s] for c in circles for s in separations]) +
                w_multilevel * lpSum([y[c] for c in circles]) +
                w_24h * M24 +
                w_extended * ME +
                sum([w_lab.get(lab, 0.05) * S[lab] for lab in selected_labs])
            )
            model += obj
            
            model += W == lpSum([standard_points[c][s] * t_standard * x[c,s] for c in circles for s in separations]) + lpSum([y[c] * t_multilevel for c in circles]) + M24 * t_24h + ME * t_extended
            model += W <= E * h_work * T_total
            
            filtered_costs = {lab: costs_lab[lab] for lab in selected_labs}
            model += daily_cost * (W / h_work) + sum([filtered_costs[lab] * S[lab] for lab in selected_labs]) <= B_total
            
            for c in circles:
                model += lpSum([x[c,s] for s in separations]) == 1
            
            model += M24 >= min_24h_total
            model += ME >= min_extended_total
            
            for lab in selected_labs:
                model += S[lab] >= min_samples[lab]
            
            for c in circles:
                model += y[c] >= min_multilevel_per_circle
            
            model += E <= max_equipment
            
            model.solve(PULP_CBC_CMD(msg=0))
            
            if model.status == 1:
                st.success("SOLUCION OPTIMA ENCONTRADA!")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Estrategia", "Costos", "Graficos"])
                
                work_hours = int(W.varValue) if W.varValue else 0
                work_days = work_hours / h_work
                operational_cost = daily_cost * work_days
                lab_cost = sum([filtered_costs.get(lab, 0) * int(S[lab].varValue or 0) for lab in selected_labs])
                total_cost = operational_cost + lab_cost
                
                with tab1:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Calidad", f"{value(model.objective):.4f}")
                    col2.metric("Equipos", int(E.varValue or 0))
                    col3.metric("Horas", work_hours)
                    col4.metric("Costo", f"${total_cost:,.0f}")
                
                with tab2:
                    strategy_data = []
                    for circle in circles:
                        for sep in separations:
                            if x[circle, sep].varValue == 1:
                                n_std = standard_points[circle][sep]
                                n_multi = int(y[circle].varValue or 0)
                                strategy_data.append({
                                    'Circulo': circle,
                                    'Separacion': sep,
                                    'Estandar': n_std,
                                    'Multinivel': n_multi,
                                    'Total': n_std + n_multi
                                })
                    if strategy_data:
                        st.dataframe(pd.DataFrame(strategy_data), use_container_width=True)
                
                with tab3:
                    margin = B_total - total_cost
                    col1, col2 = st.columns(2)
                    col1.metric("Costo Operativo", f"${operational_cost:,.0f}")
                    col2.metric("Costo Laboratorio", f"${lab_cost:,.0f}")
                    col1.metric("Presupuesto", f"${B_total:,.0f}")
                    col2.metric("Margen", f"${margin:,.0f}")
                
                with tab4:
                    fig1 = go.Figure(data=[go.Pie(labels=['Operativo', 'Laboratorio'], values=[operational_cost, lab_cost])])
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    if lab_cost > 0:
                        lab_summary = {lab: int(S[lab].varValue or 0) for lab in selected_labs}
                        fig2 = px.bar(x=list(lab_summary.keys()), y=list(lab_summary.values()), title="Muestras")
                        st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error(f"No se encontro solucion. Estado: {model.status}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info("Optimizador de Muestreo v1.0")
