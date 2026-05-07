import io
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pulp
import streamlit as st


st.set_page_config(
    page_title="Optimizador de muestreo MILP",
    page_icon="🧪",
    layout="wide",
)


DEFAULT_SEPARATIONS = [100, 150, 200, 300, 500]
DEFAULT_LAB_TYPES = [
    "Mineralógico",
    "Biogeoquímico",
    "Cromatografía",
    "Deuterio",
    "Helio",
]


@st.cache_data
def load_example_data() -> pd.DataFrame:
    try:
        df = pd.read_excel("Resultados_Optimizacion_Muestreo.xlsx", sheet_name="Estrategia")
        df = df[["Círculo", "Perímetro (m)"]].dropna()
        df["Perímetro (m)"] = pd.to_numeric(df["Perímetro (m)"], errors="coerce")
        df = df.dropna()
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(
            {
                "Círculo": [f"C{i}" for i in range(1, 6)],
                "Perímetro (m)": [113, 164, 702, 1393, 221],
            }
        )


def points_from_perimeter(perimeter: float, separation: float) -> int:
    """Número de puntos estándar calculado como ceil(perímetro/separación), mínimo 1."""
    if separation <= 0:
        return 1
    return max(1, int(math.ceil(perimeter / separation)))


def solve_model(
    circles_df: pd.DataFrame,
    separations: List[int],
    lab_types: List[str],
    total_time_h: float,
    hours_per_day: float,
    budget: float,
    daily_cost: float,
    t_standard: float,
    t_multilevel: float,
    t_24h: float,
    t_extended: float,
    min_multilevel_per_circle: int,
    min_24h: int,
    min_extended: int,
    min_lab_samples: Dict[str, int],
    max_teams: int,
    w_standard: float,
    w_multilevel: float,
    w_24h: float,
    w_extended: float,
    w_lab: Dict[str, float],
    lab_costs: Dict[str, float],
) -> Tuple[str, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    circles = circles_df["Círculo"].astype(str).tolist()
    perimeter = dict(zip(circles, circles_df["Perímetro (m)"].astype(float)))

    n_points = {
        (i, s): points_from_perimeter(perimeter[i], s)
        for i in circles
        for s in separations
    }

    model = pulp.LpProblem("Optimizacion_Muestreo", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("usar_separacion", (circles, separations), lowBound=0, upBound=1, cat="Binary")
    m = pulp.LpVariable.dicts("puntos_multinivel", circles, lowBound=0, cat="Integer")
    teams = pulp.LpVariable("equipos", lowBound=1, upBound=max_teams, cat="Integer")
    h_total = pulp.LpVariable("horas_hombre", lowBound=0, cat="Continuous")
    y_24h = pulp.LpVariable("mediciones_24h", lowBound=0, cat="Integer")
    y_ext = pulp.LpVariable("mediciones_extendidas", lowBound=0, cat="Integer")
    lab = pulp.LpVariable.dicts("muestras_lab", lab_types, lowBound=0, cat="Integer")

    standard_points_expr = pulp.lpSum(n_points[(i, s)] * x[i][s] for i in circles for s in separations)
    multilevel_expr = pulp.lpSum(m[i] for i in circles)
    lab_expr = pulp.lpSum(lab[k] for k in lab_types)

    model += (
        w_standard * standard_points_expr
        + w_multilevel * multilevel_expr
        + w_24h * y_24h
        + w_extended * y_ext
        + pulp.lpSum(w_lab[k] * lab[k] for k in lab_types)
    )

    field_hours_expr = (
        t_standard * standard_points_expr
        + t_multilevel * multilevel_expr
        + t_24h * y_24h
        + t_extended * y_ext
    )
    model += h_total == field_hours_expr, "R1_definicion_esfuerzo"
    model += h_total <= total_time_h * teams, "R2_capacidad_operativa"

    operation_days_expr = h_total / max(hours_per_day * 2, 1e-6)
    operation_cost_expr = operation_days_expr * daily_cost
    lab_cost_expr = pulp.lpSum(lab_costs[k] * lab[k] for k in lab_types)
    model += operation_cost_expr + lab_cost_expr <= budget, "R3_presupuesto"

    for i in circles:
        model += pulp.lpSum(x[i][s] for s in separations) == 1, f"R4_separacion_unica_{i}"
        model += m[i] >= min_multilevel_per_circle, f"R7_min_multinivel_{i}"

    model += y_24h >= min_24h, "R5_min_24h"
    model += y_ext >= min_extended, "R5_min_extendidas"

    for k in lab_types:
        model += lab[k] >= min_lab_samples[k], f"R6_min_lab_{k}"

    model += teams >= 1, "R8_min_equipos"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    status = pulp.LpStatus[model.status]

    rows = []
    if status in ["Optimal", "Feasible"]:
        for i in circles:
            selected_sep = None
            selected_points = None
            for s in separations:
                if pulp.value(x[i][s]) > 0.5:
                    selected_sep = s
                    selected_points = n_points[(i, s)]
                    break
            rows.append(
                {
                    "Círculo": i,
                    "Perímetro (m)": perimeter[i],
                    "Separación seleccionada (m)": selected_sep,
                    "Puntos estándar": selected_points,
                    "Puntos multinivel": int(round(pulp.value(m[i]))),
                    "Total puntos": selected_points + int(round(pulp.value(m[i]))),
                }
            )

    strategy_df = pd.DataFrame(rows)
    lab_df = pd.DataFrame(
        {
            "Tipo de análisis": lab_types,
            "Muestras": [int(round(pulp.value(lab[k]))) if status in ["Optimal", "Feasible"] else None for k in lab_types],
            "Costo unitario": [lab_costs[k] for k in lab_types],
            "Costo total": [
                (int(round(pulp.value(lab[k]))) * lab_costs[k]) if status in ["Optimal", "Feasible"] else None
                for k in lab_types
            ],
        }
    )

    if status in ["Optimal", "Feasible"]:
        total_h = float(pulp.value(h_total))
        operation_days = total_h / max(hours_per_day * 2, 1e-6)
        lab_total = float(sum(lab_df["Costo total"]))
        operation_total = operation_days * daily_cost
        metrics = {
            "Calidad científica": float(pulp.value(model.objective)),
            "Horas-hombre": total_h,
            "Días de operación estimados": operation_days,
            "Equipos": int(round(pulp.value(teams))),
            "Mediciones 24 h": int(round(pulp.value(y_24h))),
            "Mediciones extendidas": int(round(pulp.value(y_ext))),
            "Costo operativo": operation_total,
            "Costo laboratorio": lab_total,
            "Costo total": operation_total + lab_total,
            "Margen presupuesto": budget - operation_total - lab_total,
        }
    else:
        metrics = {}

    return status, strategy_df, lab_df, metrics


def to_excel(strategy_df: pd.DataFrame, lab_df: pd.DataFrame, metrics: Dict[str, float]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        strategy_df.to_excel(writer, index=False, sheet_name="Estrategia")
        lab_df.to_excel(writer, index=False, sheet_name="Laboratorio")
        pd.DataFrame(list(metrics.items()), columns=["Concepto", "Valor"]).to_excel(
            writer, index=False, sheet_name="Resumen"
        )

        workbook = writer.book
        money_fmt = workbook.add_format({"num_format": "$#,##0"})
        num_fmt = workbook.add_format({"num_format": "0.00"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})

        for sheet_name in ["Estrategia", "Laboratorio", "Resumen"]:
            ws = writer.sheets[sheet_name]
            ws.set_row(0, None, header_fmt)
            ws.set_column(0, 0, 18)
            ws.set_column(1, 10, 18)

        writer.sheets["Laboratorio"].set_column(2, 3, 18, money_fmt)
        writer.sheets["Resumen"].set_column(1, 1, 18, num_fmt)

    return output.getvalue()


st.title("🧪 Optimizador MILP de estrategia de muestreo")
st.caption("Modelo de programación lineal entera mixta para maximizar calidad científica bajo restricciones de tiempo, presupuesto y mínimos técnicos.")

with st.sidebar:
    st.header("1. Datos generales")
    use_example = st.checkbox("Usar perímetros del Excel de ejemplo", value=True)
    n_circles = st.number_input("Número de círculos", min_value=1, max_value=200, value=30, step=1)

    st.header("2. Separaciones permitidas")
    sep_text = st.text_input("Separaciones en metros", value=", ".join(map(str, DEFAULT_SEPARATIONS)))

    st.header("3. Tiempo y equipos")
    total_time_h = st.number_input("Tiempo total disponible por equipo (h)", min_value=1.0, value=1848.0, step=8.0)
    hours_per_day = st.number_input("Horas efectivas por día", min_value=1.0, value=8.0, step=1.0)
    max_teams = st.number_input("Máximo de equipos", min_value=1, max_value=20, value=3, step=1)
    t_standard = st.number_input("Tiempo por punto estándar (h)", min_value=0.0, value=1.0, step=0.25)
    t_multilevel = st.number_input("Tiempo por punto multinivel (h)", min_value=0.0, value=2.0, step=0.25)
    t_24h = st.number_input("Tiempo por medición 24 h (h)", min_value=0.0, value=24.0, step=1.0)
    t_extended = st.number_input("Tiempo por medición extendida (h)", min_value=0.0, value=48.0, step=1.0)

    st.header("4. Presupuesto")
    budget = st.number_input("Presupuesto total", min_value=0.0, value=400_000_000.0, step=1_000_000.0)
    daily_cost = st.number_input("Costo operativo diario", min_value=0.0, value=1_350_000.0, step=50_000.0)

    st.header("5. Requisitos mínimos")
    min_multilevel_per_circle = st.number_input("Puntos multinivel mínimos por círculo", min_value=0, value=3, step=1)
    min_24h = st.number_input("Mediciones 24 h mínimas", min_value=0, value=1, step=1)
    min_extended = st.number_input("Mediciones extendidas mínimas", min_value=0, value=1, step=1)

st.subheader("A. Geometría de los círculos")

if use_example:
    base_df = load_example_data().head(int(n_circles)).copy()
    if len(base_df) < n_circles:
        extra = pd.DataFrame(
            {
                "Círculo": [f"C{i}" for i in range(len(base_df) + 1, int(n_circles) + 1)],
                "Perímetro (m)": np.repeat(float(base_df["Perímetro (m)"].mean()), int(n_circles) - len(base_df)),
            }
        )
        base_df = pd.concat([base_df, extra], ignore_index=True)
else:
    base_df = pd.DataFrame({"Círculo": [f"C{i}" for i in range(1, int(n_circles) + 1)], "Perímetro (m)": [100.0] * int(n_circles)})

circles_df = st.data_editor(
    base_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={"Perímetro (m)": st.column_config.NumberColumn(min_value=1.0, step=1.0)},
)

st.subheader("B. Análisis de laboratorio")
lab_config = pd.DataFrame(
    {
        "Tipo de análisis": DEFAULT_LAB_TYPES,
        "Mínimo de muestras": [1, 1, 1, 1, 1],
        "Costo unitario": [250_000, 350_000, 450_000, 500_000, 500_000],
        "Peso calidad": [0.25, 0.30, 0.25, 0.10, 0.10],
    }
)
lab_config = st.data_editor(
    lab_config,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Mínimo de muestras": st.column_config.NumberColumn(min_value=0, step=1),
        "Costo unitario": st.column_config.NumberColumn(min_value=0.0, step=10_000.0),
        "Peso calidad": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05),
    },
)

st.subheader("C. Pesos de calidad científica")
col1, col2, col3, col4 = st.columns(4)
with col1:
    w_standard = st.number_input("Peso punto estándar", min_value=0.0, value=1.0, step=0.1)
with col2:
    w_multilevel = st.number_input("Peso punto multinivel", min_value=0.0, value=2.0, step=0.1)
with col3:
    w_24h = st.number_input("Peso medición 24 h", min_value=0.0, value=5.0, step=0.5)
with col4:
    w_extended = st.number_input("Peso medición extendida", min_value=0.0, value=8.0, step=0.5)

try:
    separations = sorted([int(float(x.strip())) for x in sep_text.split(",") if x.strip()])
except ValueError:
    separations = DEFAULT_SEPARATIONS
    st.warning("No pude interpretar las separaciones. Se usan los valores por defecto.")

lab_config = lab_config.dropna(subset=["Tipo de análisis"])
lab_types = lab_config["Tipo de análisis"].astype(str).tolist()
min_lab_samples = dict(zip(lab_types, lab_config["Mínimo de muestras"].astype(int)))
lab_costs = dict(zip(lab_types, lab_config["Costo unitario"].astype(float)))
w_lab = dict(zip(lab_types, lab_config["Peso calidad"].astype(float)))

run = st.button("Resolver modelo", type="primary", use_container_width=True)

if run:
    if circles_df.empty or not separations or not lab_types:
        st.error("Revisa que existan círculos, separaciones y tipos de análisis.")
    else:
        status, strategy_df, lab_df, metrics = solve_model(
            circles_df=circles_df,
            separations=separations,
            lab_types=lab_types,
            total_time_h=total_time_h,
            hours_per_day=hours_per_day,
            budget=budget,
            daily_cost=daily_cost,
            t_standard=t_standard,
            t_multilevel=t_multilevel,
            t_24h=t_24h,
            t_extended=t_extended,
            min_multilevel_per_circle=int(min_multilevel_per_circle),
            min_24h=int(min_24h),
            min_extended=int(min_extended),
            min_lab_samples=min_lab_samples,
            max_teams=int(max_teams),
            w_standard=w_standard,
            w_multilevel=w_multilevel,
            w_24h=w_24h,
            w_extended=w_extended,
            w_lab=w_lab,
            lab_costs=lab_costs,
        )

        if status not in ["Optimal", "Feasible"]:
            st.error(f"El modelo no encontró solución factible. Estado: {status}")
            st.info("Prueba aumentar presupuesto/tiempo, reducir mínimos o permitir más equipos.")
        else:
            st.success(f"Solución encontrada: {status}")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Calidad científica", f"{metrics['Calidad científica']:.2f}")
            kpi2.metric("Costo total", f"${metrics['Costo total']:,.0f}")
            kpi3.metric("Margen presupuesto", f"${metrics['Margen presupuesto']:,.0f}")
            kpi4.metric("Días operación", f"{metrics['Días de operación estimados']:.1f}")

            st.markdown("### Estrategia óptima")
            st.dataframe(strategy_df, use_container_width=True)

            st.markdown("### Laboratorio")
            st.dataframe(lab_df, use_container_width=True)

            st.markdown("### Resumen")
            st.json(metrics)

            excel_bytes = to_excel(strategy_df, lab_df, metrics)
            st.download_button(
                "Descargar resultados en Excel",
                data=excel_bytes,
                file_name="resultados_optimizacion_streamlit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
else:
    st.info("Ajusta los parámetros y presiona **Resolver modelo**.")
