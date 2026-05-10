import math
import os
import re
from io import BytesIO

import numpy as np
import pandas as pd
import pulp
import streamlit as st


st.set_page_config(
    page_title="Optimización MILP-AHP para muestreo de H₂",
    layout="wide",
)

st.title("Optimización MILP-AHP / Goal Programming")
st.caption("Asignación de recursos para estrategias de muestreo de seeps de hidrógeno")

BASE_DIR = os.path.dirname(__file__)
DEFAULT_CIRCLES_XLSX = os.path.join(BASE_DIR, "Circulos.xlsx")
DEFAULT_PARAMS_XLSX = os.path.join(BASE_DIR, "Parametros.xlsx")


def _num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            x = x.replace("COP", "").replace("$", "").replace(".", "").replace(",", ".")
        return float(x)
    except Exception:
        return default


def normalize_weights(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    total = s.sum()
    if total <= 0:
        return s
    return s / total


@st.cache_data
def load_real_defaults(circles_file=None, params_file=None):
    """Carga los Excel reales si existen. Si no, usa datos mínimos de ejemplo."""
    # Círculos
    if circles_file is not None:
        circles_raw = pd.read_excel(circles_file)
    elif os.path.exists(DEFAULT_CIRCLES_XLSX):
        circles_raw = pd.read_excel(DEFAULT_CIRCLES_XLSX)
    else:
        circles_raw = pd.DataFrame({"ID": ["C1", "C2", "C3"], "Perimetro (m)": [120, 180, 250]})

    circles = circles_raw.rename(columns={"ID": "circle_id", "Perimetro (m)": "perimeter_m"})
    circles = circles[["circle_id", "perimeter_m"]].copy()
    circles["circle_id"] = circles["circle_id"].astype(str).str.strip()
    circles["perimeter_m"] = pd.to_numeric(circles["perimeter_m"], errors="coerce")
    circles = circles.dropna(subset=["circle_id", "perimeter_m"])

    # Parámetros
    if params_file is not None:
        params_raw = pd.read_excel(params_file, header=None)
    elif os.path.exists(DEFAULT_PARAMS_XLSX):
        params_raw = pd.read_excel(DEFAULT_PARAMS_XLSX, header=None)
    else:
        params_raw = pd.DataFrame()

    def value_contains(text, fallback):
        if params_raw.empty:
            return fallback
        mask = params_raw[0].astype(str).str.contains(text, case=False, na=False, regex=False)
        if mask.any():
            return _num(params_raw.loc[mask, 2].iloc[0], fallback)
        return fallback

    # Valores de costo con respaldo porque el archivo original contiene referencias externas.
    budget = value_contains("Presupuesto Total Disponible", 446_852_500)
    days = value_contains("Tiempo total disponible para el muestreo", 60)
    hours_per_day = value_contains("Horas efectivas", 8)
    max_teams = int(value_contains("Cantidad de equipos/sensores disponibles", 2))
    daily_field_cost = value_contains("Logística de Muestreo", 1_513_000)
    if daily_field_cost <= 0:
        daily_field_cost = 1_513_000

    # Separaciones permitidas. El archivo real no trae este conjunto, por eso queda editable.
    separations = pd.DataFrame({"separation_m": [25.0, 50.0, 100.0, 200.0]})

    # Estrategias de campo basadas en los tiempos y pesos AHP reales.
    field = pd.DataFrame(
        {
            "field_code": ["STD", "ML", "M24", "EXT"],
            "field_type": [
                "Muestreo puntual estándar",
                "Muestreo multinivel",
                "Medición 24 horas",
                "Medición extendida 5 días",
            ],
            "time_h_per_sample": [
                value_contains("muestreo puntual estándar", 1),
                value_contains("muestreo multinivel", 3),
                value_contains("medición de 24 horas", 25),
                value_contains("medición extendida", 121),
            ],
            "min_samples": [
                value_contains("Candidatos de muestreo - Teledetección", 69),
                value_contains("Puntos multinivel mínimos", 3),
                value_contains("Mediciones de 24 horas mínimas", 10),
                value_contains("Mediciones extendidas mínimas", 2),
            ],
            "ahp_weight_raw": [
                value_contains("Calidad de un punto de muestreo estándar", 0.2417),
                value_contains("Calidad de un punto multinivel", 0.2029),
                value_contains("Calidad de una medición de 24 horas", 0.1459),
                value_contains("Calidad de una medición extendida", 0.0523),
            ],
        }
    )
    field["ahp_local_weight"] = normalize_weights(field["ahp_weight_raw"])

    lab = pd.DataFrame(
        {
            "analysis_code": ["GC", "D", "HE", "MIN", "BIO"],
            "analysis_type": [
                "Cromatografía de gas",
                "Isotopía de Deuterio",
                "Isotopía de Helio",
                "Caracterización mineralógica",
                "Biogeoquímica",
            ],
            "unit_cost": [1_190_000, 5_724_852, 5_724_852, 1_228_080, 3_015_249.37],
            "min_samples": [
                value_contains("Muestras mínimas para cromatografia", 15),
                value_contains("Muestras mínimas para isotopia de Deuterio", 5),
                value_contains("Muestras mínimas para isotopia de Helio", 5),
                value_contains("Muestras mínimas para caracterizacion mineralogica", 15),
                value_contains("Muestras mínimas para biogeoquimica", 15),
            ],
            "max_samples": [1000, 1000, 1000, 1000, 1000],
            "ahp_weight_raw": [
                value_contains("Calidad de cromatografia", 0.1511),
                value_contains("Calidad de isotopia de Deuterio", 0.0897),
                value_contains("Calidad de isotopia de Helio", 0.0438),
                value_contains("Calidad de caracterizacion mineralogica", 0.0399),
                value_contains("Calidad de biogeoquimica", 0.0328),
            ],
        }
    )
    lab["ahp_local_weight"] = normalize_weights(lab["ahp_weight_raw"])

    field_weight_raw = field["ahp_weight_raw"].sum()
    lab_weight_raw = lab["ahp_weight_raw"].sum()
    total_raw = field_weight_raw + lab_weight_raw
    w_field = field_weight_raw / total_raw if total_raw > 0 else 0.6
    w_lab = lab_weight_raw / total_raw if total_raw > 0 else 0.4

    params = {
        "budget": float(budget),
        "days": float(days),
        "hours_per_day": float(hours_per_day),
        "daily_field_cost": float(daily_field_cost),
        "max_teams": int(max_teams),
        "w_field": float(w_field),
        "w_lab": float(w_lab),
    }
    return circles, separations, field, lab, params


def clean_inputs(circles, separations, field, lab):
    circles = circles.copy()
    separations = separations.copy()
    field = field.copy()
    lab = lab.copy()

    circles["circle_id"] = circles["circle_id"].astype(str).str.strip()
    field["field_code"] = field["field_code"].astype(str).str.strip()
    field["field_type"] = field["field_type"].astype(str).str.strip()
    lab["analysis_code"] = lab["analysis_code"].astype(str).str.strip()
    lab["analysis_type"] = lab["analysis_type"].astype(str).str.strip()

    circles["perimeter_m"] = pd.to_numeric(circles["perimeter_m"], errors="coerce")
    separations["separation_m"] = pd.to_numeric(separations["separation_m"], errors="coerce")
    for col in ["time_h_per_sample", "min_samples", "ahp_weight_raw"]:
        field[col] = pd.to_numeric(field[col], errors="coerce").fillna(0)
    for col in ["unit_cost", "min_samples", "max_samples", "ahp_weight_raw"]:
        lab[col] = pd.to_numeric(lab[col], errors="coerce").fillna(0)

    field["ahp_local_weight"] = normalize_weights(field["ahp_weight_raw"])
    lab["ahp_local_weight"] = normalize_weights(lab["ahp_weight_raw"])

    circles = circles.dropna(subset=["circle_id", "perimeter_m"])
    circles = circles[circles["perimeter_m"] > 0]
    separations = separations.dropna(subset=["separation_m"])
    separations = separations[separations["separation_m"] > 0]
    field = field[(field["field_code"] != "") & (field["time_h_per_sample"] > 0)]
    lab = lab[(lab["analysis_code"] != "") & (lab["unit_cost"] >= 0) & (lab["max_samples"] >= lab["min_samples"])]

    if circles.empty or separations.empty or field.empty or lab.empty:
        raise ValueError("Revisa los datos: no puede haber tablas vacías ni valores negativos/incorrectos.")
    if field["ahp_local_weight"].sum() <= 0 or lab["ahp_local_weight"].sum() <= 0:
        raise ValueError("Los pesos AHP deben sumar más que cero.")
    if circles["circle_id"].duplicated().any():
        raise ValueError("Los IDs de círculos deben ser únicos.")
    if field["field_code"].duplicated().any() or lab["analysis_code"].duplicated().any():
        raise ValueError("Los códigos de campo y laboratorio deben ser únicos.")

    return circles, separations, field, lab


def build_model(circles, separations, field, lab, params):
    circle_ids = circles["circle_id"].tolist()
    sep_vals = separations["separation_m"].tolist()
    field_codes = field["field_code"].tolist()
    lab_codes = lab["analysis_code"].tolist()

    field_name = dict(zip(field["field_code"], field["field_type"]))
    lab_name = dict(zip(lab["analysis_code"], lab["analysis_type"]))

    P = dict(zip(circles["circle_id"], circles["perimeter_m"]))
    n_points = {(i, s): int(math.ceil(P[i] / s)) for i in circle_ids for s in sep_vals}
    t = dict(zip(field["field_code"], field["time_h_per_sample"]))
    wf = dict(zip(field["field_code"], field["ahp_local_weight"]))
    min_field = dict(zip(field["field_code"], field["min_samples"]))
    ca = dict(zip(lab["analysis_code"], lab["unit_cost"]))
    wa = dict(zip(lab["analysis_code"], lab["ahp_local_weight"]))
    min_lab = dict(zip(lab["analysis_code"], lab["min_samples"]))
    max_lab = dict(zip(lab["analysis_code"], lab["max_samples"]))

    B = float(params["budget"])
    D = float(params["days"])
    H = float(params["hours_per_day"])
    c_day = float(params["daily_field_cost"])
    max_teams = int(params["max_teams"])
    W_field = float(params["w_field"])
    W_lab = float(params["w_lab"])
    couple_lab_to_field = bool(params.get("couple_lab_to_field", True))
    c_hour = c_day / H

    m = pulp.LpProblem("MILP_AHP_Goal_Programming", pulp.LpMinimize)

    y = pulp.LpVariable.dicts("activate", (circle_ids, field_codes), cat="Binary")
    x = pulp.LpVariable.dicts("sep_choice", (circle_ids, field_codes, sep_vals), cat="Binary")
    q_lab = pulp.LpVariable.dicts("lab_samples", lab_codes, lowBound=0, cat="Integer")
    teams = pulp.LpVariable("teams", lowBound=1, upBound=max_teams, cat="Integer")

    g_field = pulp.LpVariable.dicts("field_cost", field_codes, lowBound=0, cat="Continuous")
    g_lab = pulp.LpVariable.dicts("lab_cost", lab_codes, lowBound=0, cat="Continuous")
    total_used = pulp.LpVariable("total_budget_used", lowBound=0, cat="Continuous")
    total_field = pulp.LpVariable("total_field_cost", lowBound=0, cat="Continuous")
    total_lab = pulp.LpVariable("total_lab_cost", lowBound=0, cat="Continuous")

    dev_block_field = pulp.LpVariable("dev_block_field", lowBound=0, cat="Continuous")
    dev_block_lab = pulp.LpVariable("dev_block_lab", lowBound=0, cat="Continuous")
    dev_field = pulp.LpVariable.dicts("dev_field", field_codes, lowBound=0, cat="Continuous")
    dev_lab = pulp.LpVariable.dicts("dev_lab", lab_codes, lowBound=0, cat="Continuous")

    for i in circle_ids:
        safe_i = re.sub(r"\W+", "_", str(i))
        m += pulp.lpSum(y[i][k] for k in field_codes) >= 1, f"at_least_one_activity_{safe_i}"
        for k in field_codes:
            m += pulp.lpSum(x[i][k][s] for s in sep_vals) == y[i][k], f"one_sep_if_active_{safe_i}_{k}"

    field_qty_expr = {}
    for k in field_codes:
        field_qty_expr[k] = pulp.lpSum(n_points[(i, s)] * x[i][k][s] for i in circle_ids for s in sep_vals)
        m += field_qty_expr[k] >= min_field[k], f"min_field_{k}"
        m += g_field[k] == c_hour * t[k] * field_qty_expr[k], f"field_cost_{k}"

    total_field_samples = pulp.lpSum(field_qty_expr[k] for k in field_codes)

    for a in lab_codes:
        m += q_lab[a] >= min_lab[a], f"min_lab_samples_{a}"
        m += q_lab[a] <= max_lab[a], f"max_lab_samples_{a}"
        if couple_lab_to_field:
            m += q_lab[a] <= total_field_samples, f"lab_coupled_to_field_{a}"
        m += g_lab[a] == ca[a] * q_lab[a], f"lab_cost_{a}"

    m += total_field == pulp.lpSum(g_field[k] for k in field_codes), "total_field"
    m += total_lab == pulp.lpSum(g_lab[a] for a in lab_codes), "total_lab"
    m += total_used == total_field + total_lab, "total_used"
    m += total_used <= B, "budget_limit"

    total_time = pulp.lpSum(t[k] * field_qty_expr[k] for k in field_codes)
    m += total_time <= D * H * teams, "time_capacity"

    m += total_field - W_field * total_used <= dev_block_field, "block_field_pos"
    m += W_field * total_used - total_field <= dev_block_field, "block_field_neg"
    m += total_lab - W_lab * total_used <= dev_block_lab, "block_lab_pos"
    m += W_lab * total_used - total_lab <= dev_block_lab, "block_lab_neg"

    for k in field_codes:
        m += g_field[k] - wf[k] * total_field <= dev_field[k], f"field_dev_pos_{k}"
        m += wf[k] * total_field - g_field[k] <= dev_field[k], f"field_dev_neg_{k}"

    for a in lab_codes:
        m += g_lab[a] - wa[a] * total_lab <= dev_lab[a], f"lab_dev_pos_{a}"
        m += wa[a] * total_lab - g_lab[a] <= dev_lab[a], f"lab_dev_neg_{a}"

    return {
        "model": m,
        "sets": (circle_ids, sep_vals, field_codes, lab_codes),
        "names": {"field": field_name, "lab": lab_name},
        "vars": {
            "x": x,
            "y": y,
            "q_lab": q_lab,
            "teams": teams,
            "g_field": g_field,
            "g_lab": g_lab,
            "total_used": total_used,
            "total_field": total_field,
            "total_lab": total_lab,
            "dev_block_field": dev_block_field,
            "dev_block_lab": dev_block_lab,
            "dev_field": dev_field,
            "dev_lab": dev_lab,
        },
        "n_points": n_points,
        "field_qty_expr": field_qty_expr,
        "total_field_samples": total_field_samples,
        "total_time_expr": total_time,
    }


def solve_relaxed_minimums(circles, separations, field, lab, params, time_limit_sec=30):
    """Calcula mínimos auxiliares relajando presupuesto y capacidad de tiempo.

    Sirve para explicar por qué algunas separaciones vuelven infactible el modelo.
    No reemplaza la optimización lexicográfica final.
    """
    relaxed_params = dict(params)
    relaxed_params["budget"] = 1e15
    relaxed_params["days"] = 1e9

    objects_time = build_model(circles, separations, field, lab, relaxed_params)
    m_time = objects_time["model"]
    m_time.sense = pulp.LpMinimize
    m_time.setObjective(objects_time["total_time_expr"])
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    status_time = m_time.solve(solver)
    min_time = None
    if pulp.LpStatus[status_time] == "Optimal":
        min_time = float(pulp.value(objects_time["total_time_expr"]))

    objects_cost = build_model(circles, separations, field, lab, relaxed_params)
    m_cost = objects_cost["model"]
    m_cost.sense = pulp.LpMinimize
    m_cost.setObjective(objects_cost["vars"]["total_used"])
    status_cost = m_cost.solve(solver)
    min_cost = None
    if pulp.LpStatus[status_cost] == "Optimal":
        min_cost = float(pulp.value(objects_cost["vars"]["total_used"]))

    return {
        "status_time": pulp.LpStatus[status_time],
        "min_time_h": min_time,
        "status_cost": pulp.LpStatus[status_cost],
        "min_cost": min_cost,
    }


def solve_lexicographic(objects, time_limit_sec=60):
    m = objects["model"]
    _, _, field_codes, lab_codes = objects["sets"]
    v = objects["vars"]
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    # Tolerancia lexicográfica en COP.
    # Usar 1e-5 COP es demasiado estricto para un MILP con costos grandes
    # e integridad/discretización; CBC puede declarar infeasible por tolerancias numéricas.
    def lex_tol(value):
        return max(1000.0, 1e-6 * abs(float(value)))

    z1_expr = v["dev_block_field"] + v["dev_block_lab"]
    m.sense = pulp.LpMinimize
    m.setObjective(z1_expr)
    status1 = m.solve(solver)
    if pulp.LpStatus[status1] != "Optimal":
        raise RuntimeError(f"Etapa 1 no factible u óptimo no encontrado: {pulp.LpStatus[status1]}")
    z1 = pulp.value(z1_expr)
    m += z1_expr <= z1 + lex_tol(z1), "fix_Z1"

    z2_expr = pulp.lpSum(v["dev_field"][k] for k in field_codes) + pulp.lpSum(v["dev_lab"][a] for a in lab_codes)
    m.sense = pulp.LpMinimize
    m.setObjective(z2_expr)
    status2 = m.solve(solver)
    if pulp.LpStatus[status2] != "Optimal":
        raise RuntimeError(f"Etapa 2 no factible u óptimo no encontrado: {pulp.LpStatus[status2]}")
    z2 = pulp.value(z2_expr)
    m += z2_expr <= z2 + lex_tol(z2), "fix_Z2"

    m.sense = pulp.LpMaximize
    m.setObjective(v["total_used"])
    status3 = m.solve(solver)
    if pulp.LpStatus[status3] != "Optimal":
        raise RuntimeError(f"Etapa 3 no factible u óptimo no encontrado: {pulp.LpStatus[status3]}")

    return {
        "status": pulp.LpStatus[status3],
        "z1": float(z1),
        "z2": float(z2),
        "objective_budget_used": float(pulp.value(v["total_used"])),
    }


def extract_results(objects):
    circle_ids, sep_vals, field_codes, lab_codes = objects["sets"]
    names = objects["names"]
    v = objects["vars"]
    n_points = objects["n_points"]

    rows = []
    for i in circle_ids:
        for k in field_codes:
            if pulp.value(v["y"][i][k]) and pulp.value(v["y"][i][k]) > 0.5:
                chosen_sep = None
                chosen_points = None
                for s in sep_vals:
                    if pulp.value(v["x"][i][k][s]) and pulp.value(v["x"][i][k][s]) > 0.5:
                        chosen_sep = s
                        chosen_points = n_points[(i, s)]
                        break
                rows.append(
                    {
                        "circle_id": i,
                        "field_code": k,
                        "field_type": names["field"][k],
                        "separation_m": chosen_sep,
                        "n_points": chosen_points,
                    }
                )
    field_plan = pd.DataFrame(rows)

    field_costs = pd.DataFrame(
        {
            "field_code": field_codes,
            "field_type": [names["field"][k] for k in field_codes],
            "field_samples": [round(pulp.value(objects["field_qty_expr"][k])) for k in field_codes],
            "field_cost": [pulp.value(v["g_field"][k]) for k in field_codes],
            "field_deviation": [pulp.value(v["dev_field"][k]) for k in field_codes],
        }
    )
    lab_plan = pd.DataFrame(
        {
            "analysis_code": lab_codes,
            "analysis_type": [names["lab"][a] for a in lab_codes],
            "lab_samples": [int(round(pulp.value(v["q_lab"][a]))) for a in lab_codes],
            "lab_cost": [pulp.value(v["g_lab"][a]) for a in lab_codes],
            "lab_deviation": [pulp.value(v["dev_lab"][a]) for a in lab_codes],
        }
    )
    summary = pd.DataFrame(
        {
            "metric": [
                "Equipos seleccionados",
                "Gasto total usado",
                "Gasto campo",
                "Gasto laboratorio",
                "Muestras/puntos de campo",
                "Desviación bloque campo",
                "Desviación bloque laboratorio",
            ],
            "value": [
                pulp.value(v["teams"]),
                pulp.value(v["total_used"]),
                pulp.value(v["total_field"]),
                pulp.value(v["total_lab"]),
                pulp.value(objects["total_field_samples"]),
                pulp.value(v["dev_block_field"]),
                pulp.value(v["dev_block_lab"]),
            ],
        }
    )
    return summary, field_plan, field_costs, lab_plan


def to_excel(sheets):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    output.seek(0)
    return output


def money(x):
    return f"COP {x:,.0f}"


with st.expander("Cargar otros Excel", expanded=False):
    uploaded_circles = st.file_uploader("Circulos.xlsx", type=["xlsx"], key="uploaded_circles")
    uploaded_params = st.file_uploader("Parametros.xlsx", type=["xlsx"], key="uploaded_params")
    st.caption("Si no cargas archivos, la app usa los Excel reales incluidos en el repositorio.")

circles0, separations0, field0, lab0, params0 = load_real_defaults(uploaded_circles, uploaded_params)

with st.sidebar:
    st.header("Parámetros globales")
    budget = st.number_input("Presupuesto total (COP)", min_value=0.0, value=params0["budget"], step=1_000_000.0, format="%.0f")
    days = st.number_input("Días disponibles", min_value=0.1, value=params0["days"], step=1.0)
    hours_per_day = st.number_input("Horas efectivas por día", min_value=0.1, value=params0["hours_per_day"], step=0.5)
    daily_field_cost = st.number_input("Costo diario equipo de campo (COP/día)", min_value=0.0, value=params0["daily_field_cost"], step=100_000.0, format="%.0f")
    max_teams = st.number_input("Máximo de equipos/sensores", min_value=1, value=int(params0["max_teams"]), step=1)
    st.divider()
    st.header("Pesos AHP primer nivel")
    w_field_raw = st.number_input("Peso bloque campo", min_value=0.0, value=params0["w_field"], step=0.01, format="%.4f")
    w_lab_raw = st.number_input("Peso bloque laboratorio", min_value=0.0, value=params0["w_lab"], step=0.01, format="%.4f")
    total_w = w_field_raw + w_lab_raw
    if total_w > 0:
        w_field = w_field_raw / total_w
        w_lab = w_lab_raw / total_w
    else:
        w_field = 0.5
        w_lab = 0.5
    st.info(f"Pesos normalizados: campo={w_field:.3f}, laboratorio={w_lab:.3f}")
    st.divider()
    couple_lab_to_field = st.checkbox("Acoplar análisis al total de muestras de campo", value=True)
    time_limit_sec = st.slider("Límite de tiempo del solver (s)", 5, 300, 60)

st.subheader("1. Datos de entrada")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Círculos", len(circles0))
c2.metric("Presupuesto", money(budget))
c3.metric("Días", f"{days:g}")
c4.metric("Equipos máx.", int(max_teams))

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### Círculos")
    circles = st.data_editor(circles0, num_rows="dynamic", use_container_width=True, key="circles")
with col2:
    st.markdown("### Separaciones permitidas")
    separations = st.data_editor(separations0, num_rows="dynamic", use_container_width=True, key="separations")
    st.caption("Este conjunto no venía en los Excel reales. La información queda cargada, pero puedes editarla antes de ejecutar la optimización.")
    st.info("Tip: separaciones muy pequeñas, por ejemplo 2–10 m, aumentan mucho los puntos `ceil(perímetro/separación)` y pueden volver infactible el modelo por tiempo.")

col3, col4 = st.columns(2)
with col3:
    st.markdown("### Estrategias de campo")
    field = st.data_editor(field0, num_rows="dynamic", use_container_width=True, key="field")
with col4:
    st.markdown("### Análisis de laboratorio")
    lab = st.data_editor(lab0, num_rows="dynamic", use_container_width=True, key="lab")

st.subheader("2. Diagnóstico y resolución")

col_diag, col_run = st.columns([1, 1])
with col_diag:
    run_diagnostic = st.button("Diagnóstico de factibilidad", type="secondary")
with col_run:
    run_optimization = st.button("Ejecutar optimización", type="primary")

if run_diagnostic:
    try:
        circles_c, separations_c, field_c, lab_c = clean_inputs(circles, separations, field, lab)
        params = {
            "budget": budget,
            "days": days,
            "hours_per_day": hours_per_day,
            "daily_field_cost": daily_field_cost,
            "max_teams": max_teams,
            "w_field": w_field,
            "w_lab": w_lab,
            "couple_lab_to_field": couple_lab_to_field,
        }
        diag = solve_relaxed_minimums(circles_c, separations_c, field_c, lab_c, params, time_limit_sec=min(time_limit_sec, 60))
        capacity_h = float(days) * float(hours_per_day) * int(max_teams)
        st.markdown("### Diagnóstico rápido")
        c1, c2, c3 = st.columns(3)
        c1.metric("Capacidad real", f"{capacity_h:,.0f} h")
        c2.metric("Tiempo mínimo estimado", "—" if diag["min_time_h"] is None else f"{diag['min_time_h']:,.0f} h")
        c3.metric("Costo mínimo estimado", "—" if diag["min_cost"] is None else money(diag["min_cost"]))
        if diag["min_time_h"] is not None and diag["min_time_h"] > capacity_h:
            st.error(
                "Con estas separaciones, el tiempo mínimo requerido supera la capacidad disponible. "
                "Prueba separaciones mayores, más días, más equipos, o relaja mínimos/actividades obligatorias."
            )
        elif diag["min_cost"] is not None and diag["min_cost"] > budget:
            st.error(
                "Con estas entradas, el costo mínimo requerido supera el presupuesto. "
                "Prueba aumentar presupuesto o reducir mínimos/costos."
            )
        else:
            st.success("El diagnóstico básico no detectó una violación de tiempo o presupuesto. Puedes ejecutar la optimización completa.")
    except Exception as exc:
        st.error(str(exc))
        try:
            circles_c, separations_c, field_c, lab_c = clean_inputs(circles, separations, field, lab)
            params = {
                "budget": budget,
                "days": days,
                "hours_per_day": hours_per_day,
                "daily_field_cost": daily_field_cost,
                "max_teams": max_teams,
                "w_field": w_field,
                "w_lab": w_lab,
                "couple_lab_to_field": couple_lab_to_field,
            }
            diag = solve_relaxed_minimums(circles_c, separations_c, field_c, lab_c, params, time_limit_sec=min(time_limit_sec, 60))
            capacity_h = float(days) * float(hours_per_day) * int(max_teams)
            st.markdown("#### Posible causa")
            if diag["min_time_h"] is not None:
                st.write(f"Capacidad real: **{capacity_h:,.0f} h**. Tiempo mínimo requerido con estas separaciones: **{diag['min_time_h']:,.0f} h**.")
                if diag["min_time_h"] > capacity_h:
                    st.warning("El modelo no alcanza la capacidad de tiempo. Usa separaciones más grandes o aumenta días/equipos.")
            if diag["min_cost"] is not None:
                st.write(f"Presupuesto real: **{money(budget)}**. Costo mínimo requerido: **{money(diag['min_cost'])}**.")
                if diag["min_cost"] > budget:
                    st.warning("El modelo no alcanza el presupuesto mínimo requerido. Aumenta presupuesto o reduce mínimos/costos.")
        except Exception:
            pass

if run_optimization:
    try:
        circles_c, separations_c, field_c, lab_c = clean_inputs(circles, separations, field, lab)
        params = {
            "budget": budget,
            "days": days,
            "hours_per_day": hours_per_day,
            "daily_field_cost": daily_field_cost,
            "max_teams": max_teams,
            "w_field": w_field,
            "w_lab": w_lab,
            "couple_lab_to_field": couple_lab_to_field,
        }
        objects = build_model(circles_c, separations_c, field_c, lab_c, params)
        result = solve_lexicographic(objects, time_limit_sec=time_limit_sec)
        summary, field_plan, field_costs, lab_plan = extract_results(objects)

        st.success(f"Estado del solver: {result['status']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Z1 desviación primer nivel", f"{result['z1']:,.0f}")
        c2.metric("Z2 desviación interna", f"{result['z2']:,.0f}")
        c3.metric("Presupuesto usado", money(result["objective_budget_used"]))

        st.markdown("### Resumen")
        st.dataframe(summary, use_container_width=True)

        st.markdown("### Plan de muestreo de campo")
        st.dataframe(field_plan, use_container_width=True)

        st.markdown("### Costos y cantidades de campo")
        st.dataframe(field_costs, use_container_width=True)
        st.bar_chart(field_costs.set_index("field_type")[["field_cost"]])

        st.markdown("### Plan de laboratorio")
        st.dataframe(lab_plan, use_container_width=True)
        st.bar_chart(lab_plan.set_index("analysis_type")[["lab_cost"]])

        excel = to_excel(
            {
                "summary": summary,
                "field_plan": field_plan,
                "field_costs": field_costs,
                "lab_plan": lab_plan,
                "input_circles": circles_c,
                "input_field": field_c,
                "input_lab": lab_c,
            }
        )
        st.download_button(
            "Descargar resultados en Excel",
            data=excel,
            file_name="resultados_milp_ahp.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        st.error(str(exc))
        try:
            circles_c, separations_c, field_c, lab_c = clean_inputs(circles, separations, field, lab)
            params = {
                "budget": budget,
                "days": days,
                "hours_per_day": hours_per_day,
                "daily_field_cost": daily_field_cost,
                "max_teams": max_teams,
                "w_field": w_field,
                "w_lab": w_lab,
                "couple_lab_to_field": couple_lab_to_field,
            }
            diag = solve_relaxed_minimums(circles_c, separations_c, field_c, lab_c, params, time_limit_sec=min(time_limit_sec, 60))
            capacity_h = float(days) * float(hours_per_day) * int(max_teams)
            st.markdown("#### Posible causa")
            if diag["min_time_h"] is not None:
                st.write(f"Capacidad real: **{capacity_h:,.0f} h**. Tiempo mínimo requerido con estas separaciones: **{diag['min_time_h']:,.0f} h**.")
                if diag["min_time_h"] > capacity_h:
                    st.warning("El modelo no alcanza la capacidad de tiempo. Usa separaciones más grandes o aumenta días/equipos.")
            if diag["min_cost"] is not None:
                st.write(f"Presupuesto real: **{money(budget)}**. Costo mínimo requerido: **{money(diag['min_cost'])}**.")
                if diag["min_cost"] > budget:
                    st.warning("El modelo no alcanza el presupuesto mínimo requerido. Aumenta presupuesto o reduce mínimos/costos.")
        except Exception:
            pass

with st.expander("Notas del modelo implementado"):
    st.markdown(
        """
        - La cantidad de puntos por círculo y separación se calcula como `ceil(perímetro / separación)`.
        - Las filas `min_samples` se agregan como mínimos obligatorios por estrategia o análisis.
        - La etapa 1 minimiza desviaciones entre bloques AHP: campo vs. laboratorio.
        - La etapa 2 minimiza desviaciones internas dentro de campo y laboratorio, fijando el resultado de la etapa 1.
        - La etapa 3 maximiza el presupuesto utilizado, fijando las desviaciones óptimas de las etapas anteriores.
        - Para evitar falsos `Infeasible` por tolerancia numérica, el fijado lexicográfico usa una tolerancia mínima de COP 1.000.
        - La app incluye un diagnóstico auxiliar para estimar si una combinación de separaciones exige más horas o presupuesto del disponible.
        - Los pesos AHP locales se normalizan automáticamente dentro de cada bloque.
        - El archivo de parámetros original contiene algunas referencias externas. Para evitar errores en Streamlit Cloud, la app usa los valores guardados y respaldos explícitos para costos clave.
        """
    )
