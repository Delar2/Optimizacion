import math
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


@st.cache_data
def default_data():
    circles = pd.DataFrame(
        {
            "circle_id": ["C1", "C2", "C3"],
            "perimeter_m": [120.0, 180.0, 250.0],
        }
    )
    separations = pd.DataFrame({"separation_m": [5.0, 10.0, 20.0]})
    field = pd.DataFrame(
        {
            "field_type": ["gas_suelo", "flujo", "suelo"],
            "time_h_per_sample": [0.25, 0.40, 0.30],
            "ahp_local_weight": [0.45, 0.35, 0.20],
        }
    )
    lab = pd.DataFrame(
        {
            "analysis_type": ["GC", "isotopos", "geoquimica"],
            "unit_cost": [45.0, 120.0, 65.0],
            "ahp_local_weight": [0.50, 0.30, 0.20],
            "max_samples": [1000, 1000, 1000],
        }
    )
    return circles, separations, field, lab


def normalize_weights(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    total = s.sum()
    if total <= 0:
        return s
    return s / total


def clean_inputs(circles, separations, field, lab):
    circles = circles.copy()
    separations = separations.copy()
    field = field.copy()
    lab = lab.copy()

    circles["circle_id"] = circles["circle_id"].astype(str).str.strip()
    field["field_type"] = field["field_type"].astype(str).str.strip()
    lab["analysis_type"] = lab["analysis_type"].astype(str).str.strip()

    circles["perimeter_m"] = pd.to_numeric(circles["perimeter_m"], errors="coerce")
    separations["separation_m"] = pd.to_numeric(separations["separation_m"], errors="coerce")
    field["time_h_per_sample"] = pd.to_numeric(field["time_h_per_sample"], errors="coerce")
    field["ahp_local_weight"] = normalize_weights(field["ahp_local_weight"])
    lab["unit_cost"] = pd.to_numeric(lab["unit_cost"], errors="coerce")
    lab["ahp_local_weight"] = normalize_weights(lab["ahp_local_weight"])
    lab["max_samples"] = pd.to_numeric(lab["max_samples"], errors="coerce").fillna(1000).astype(int)

    circles = circles.dropna(subset=["circle_id", "perimeter_m"])
    circles = circles[circles["perimeter_m"] > 0]
    separations = separations.dropna(subset=["separation_m"])
    separations = separations[separations["separation_m"] > 0]
    field = field.dropna(subset=["field_type", "time_h_per_sample"])
    field = field[field["time_h_per_sample"] > 0]
    lab = lab.dropna(subset=["analysis_type", "unit_cost"])
    lab = lab[(lab["unit_cost"] >= 0) & (lab["max_samples"] >= 0)]

    if circles.empty or separations.empty or field.empty or lab.empty:
        raise ValueError("Revisa los datos: no puede haber tablas vacías ni valores negativos/incorrectos.")
    if field["ahp_local_weight"].sum() <= 0 or lab["ahp_local_weight"].sum() <= 0:
        raise ValueError("Los pesos AHP locales deben sumar más que cero.")

    return circles, separations, field, lab


def build_model(circles, separations, field, lab, params):
    circle_ids = circles["circle_id"].tolist()
    sep_vals = separations["separation_m"].tolist()
    field_types = field["field_type"].tolist()
    lab_types = lab["analysis_type"].tolist()

    P = dict(zip(circles["circle_id"], circles["perimeter_m"]))
    n_points = {(i, s): int(math.ceil(P[i] / s)) for i in circle_ids for s in sep_vals}
    t = dict(zip(field["field_type"], field["time_h_per_sample"]))
    wf = dict(zip(field["field_type"], field["ahp_local_weight"]))
    ca = dict(zip(lab["analysis_type"], lab["unit_cost"]))
    wa = dict(zip(lab["analysis_type"], lab["ahp_local_weight"]))
    max_lab = dict(zip(lab["analysis_type"], lab["max_samples"]))

    B = float(params["budget"])
    D = float(params["days"])
    H = float(params["hours_per_day"])
    c_day = float(params["daily_field_cost"])
    max_teams = int(params["max_teams"])
    W_field = float(params["w_field"])
    W_lab = float(params["w_lab"])
    c_hour = c_day / H

    m = pulp.LpProblem("MILP_AHP_Goal_Programming", pulp.LpMinimize)

    y = pulp.LpVariable.dicts("activate", (circle_ids, field_types), cat="Binary")
    x = pulp.LpVariable.dicts("sep_choice", (circle_ids, field_types, sep_vals), cat="Binary")
    q_lab = pulp.LpVariable.dicts("lab_samples", lab_types, lowBound=0, cat="Integer")
    teams = pulp.LpVariable("teams", lowBound=1, upBound=max_teams, cat="Integer")

    g_field = pulp.LpVariable.dicts("field_cost", field_types, lowBound=0, cat="Continuous")
    g_lab = pulp.LpVariable.dicts("lab_cost", lab_types, lowBound=0, cat="Continuous")
    total_used = pulp.LpVariable("total_budget_used", lowBound=0, cat="Continuous")
    total_field = pulp.LpVariable("total_field_cost", lowBound=0, cat="Continuous")
    total_lab = pulp.LpVariable("total_lab_cost", lowBound=0, cat="Continuous")

    dev_block_field = pulp.LpVariable("dev_block_field", lowBound=0, cat="Continuous")
    dev_block_lab = pulp.LpVariable("dev_block_lab", lowBound=0, cat="Continuous")
    dev_field = pulp.LpVariable.dicts("dev_field", field_types, lowBound=0, cat="Continuous")
    dev_lab = pulp.LpVariable.dicts("dev_lab", lab_types, lowBound=0, cat="Continuous")

    # Activación y elección geométrica
    for i in circle_ids:
        m += pulp.lpSum(y[i][k] for k in field_types) >= 1, f"at_least_one_activity_{i}"
        for k in field_types:
            m += pulp.lpSum(x[i][k][s] for s in sep_vals) == y[i][k], f"one_sep_if_active_{i}_{k}"

    # Costos de campo por tipo
    field_qty_expr = {}
    for k in field_types:
        field_qty_expr[k] = pulp.lpSum(n_points[(i, s)] * x[i][k][s] for i in circle_ids for s in sep_vals)
        m += g_field[k] == c_hour * t[k] * field_qty_expr[k], f"field_cost_{k}"

    # Costos de laboratorio
    for a in lab_types:
        m += q_lab[a] <= max_lab[a], f"max_lab_samples_{a}"
        m += g_lab[a] == ca[a] * q_lab[a], f"lab_cost_{a}"

    # Agregados
    m += total_field == pulp.lpSum(g_field[k] for k in field_types), "total_field"
    m += total_lab == pulp.lpSum(g_lab[a] for a in lab_types), "total_lab"
    m += total_used == total_field + total_lab, "total_used"
    m += total_used <= B, "budget_limit"

    # Capacidad operativa
    total_time = pulp.lpSum(t[k] * field_qty_expr[k] for k in field_types)
    m += total_time <= D * H * teams, "time_capacity"

    # Desviaciones absolutas de primer nivel
    m += total_field - W_field * total_used <= dev_block_field, "block_field_pos"
    m += W_field * total_used - total_field <= dev_block_field, "block_field_neg"
    m += total_lab - W_lab * total_used <= dev_block_lab, "block_lab_pos"
    m += W_lab * total_used - total_lab <= dev_block_lab, "block_lab_neg"

    # Desviaciones internas campo
    for k in field_types:
        m += g_field[k] - wf[k] * total_field <= dev_field[k], f"field_dev_pos_{k}"
        m += wf[k] * total_field - g_field[k] <= dev_field[k], f"field_dev_neg_{k}"

    # Desviaciones internas laboratorio
    for a in lab_types:
        m += g_lab[a] - wa[a] * total_lab <= dev_lab[a], f"lab_dev_pos_{a}"
        m += wa[a] * total_lab - g_lab[a] <= dev_lab[a], f"lab_dev_neg_{a}"

    objects = {
        "model": m,
        "sets": (circle_ids, sep_vals, field_types, lab_types),
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
    }
    return objects


def solve_lexicographic(objects, time_limit_sec=60):
    m = objects["model"]
    circle_ids, sep_vals, field_types, lab_types = objects["sets"]
    v = objects["vars"]
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    eps = 1e-5

    z1_expr = v["dev_block_field"] + v["dev_block_lab"]
    m.setObjective(z1_expr)
    status1 = m.solve(solver)
    if pulp.LpStatus[status1] not in ["Optimal", "Not Solved"]:
        raise RuntimeError(f"Etapa 1 no factible: {pulp.LpStatus[status1]}")
    z1 = pulp.value(z1_expr)
    m += z1_expr <= z1 + eps, "fix_Z1"

    z2_expr = pulp.lpSum(v["dev_field"][k] for k in field_types) + pulp.lpSum(v["dev_lab"][a] for a in lab_types)
    m.setObjective(z2_expr)
    status2 = m.solve(solver)
    if pulp.LpStatus[status2] not in ["Optimal", "Not Solved"]:
        raise RuntimeError(f"Etapa 2 no factible: {pulp.LpStatus[status2]}")
    z2 = pulp.value(z2_expr)
    m += z2_expr <= z2 + eps, "fix_Z2"

    m.sense = pulp.LpMaximize
    m.setObjective(v["total_used"])
    status3 = m.solve(solver)
    if pulp.LpStatus[status3] not in ["Optimal", "Not Solved"]:
        raise RuntimeError(f"Etapa 3 no factible: {pulp.LpStatus[status3]}")

    return {
        "status": pulp.LpStatus[status3],
        "z1": z1,
        "z2": z2,
        "objective_budget_used": pulp.value(v["total_used"]),
    }


def extract_results(objects):
    circle_ids, sep_vals, field_types, lab_types = objects["sets"]
    v = objects["vars"]
    n_points = objects["n_points"]

    rows = []
    for i in circle_ids:
        for k in field_types:
            if pulp.value(v["y"][i][k]) > 0.5:
                chosen_sep = None
                chosen_points = None
                for s in sep_vals:
                    if pulp.value(v["x"][i][k][s]) > 0.5:
                        chosen_sep = s
                        chosen_points = n_points[(i, s)]
                        break
                rows.append(
                    {
                        "circle_id": i,
                        "field_type": k,
                        "separation_m": chosen_sep,
                        "n_points": chosen_points,
                    }
                )
    field_plan = pd.DataFrame(rows)

    field_costs = pd.DataFrame(
        {
            "field_type": field_types,
            "field_cost": [pulp.value(v["g_field"][k]) for k in field_types],
            "field_deviation": [pulp.value(v["dev_field"][k]) for k in field_types],
        }
    )
    lab_plan = pd.DataFrame(
        {
            "analysis_type": lab_types,
            "lab_samples": [int(round(pulp.value(v["q_lab"][a]))) for a in lab_types],
            "lab_cost": [pulp.value(v["g_lab"][a]) for a in lab_types],
            "lab_deviation": [pulp.value(v["dev_lab"][a]) for a in lab_types],
        }
    )
    summary = pd.DataFrame(
        {
            "metric": [
                "Equipos seleccionados",
                "Gasto total usado",
                "Gasto campo",
                "Gasto laboratorio",
                "Desviación bloque campo",
                "Desviación bloque laboratorio",
            ],
            "value": [
                pulp.value(v["teams"]),
                pulp.value(v["total_used"]),
                pulp.value(v["total_field"]),
                pulp.value(v["total_lab"]),
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


circles0, separations0, field0, lab0 = default_data()

with st.sidebar:
    st.header("Parámetros globales")
    budget = st.number_input("Presupuesto total ($)", min_value=0.0, value=15000.0, step=500.0)
    days = st.number_input("Días disponibles", min_value=0.1, value=5.0, step=0.5)
    hours_per_day = st.number_input("Horas efectivas por día", min_value=0.1, value=8.0, step=0.5)
    daily_field_cost = st.number_input("Costo diario equipo de campo ($/día)", min_value=0.0, value=800.0, step=50.0)
    max_teams = st.number_input("Máximo de equipos", min_value=1, value=3, step=1)
    st.divider()
    st.header("Pesos AHP primer nivel")
    w_field_raw = st.number_input("Peso bloque campo", min_value=0.0, value=0.60, step=0.05)
    w_lab_raw = st.number_input("Peso bloque laboratorio", min_value=0.0, value=0.40, step=0.05)
    total_w = w_field_raw + w_lab_raw
    if total_w > 0:
        w_field = w_field_raw / total_w
        w_lab = w_lab_raw / total_w
    else:
        w_field = 0.5
        w_lab = 0.5
    st.info(f"Pesos normalizados: campo={w_field:.3f}, laboratorio={w_lab:.3f}")
    time_limit_sec = st.slider("Límite de tiempo del solver (s)", 5, 300, 60)

st.subheader("1. Datos de entrada")

col1, col2 = st.columns(2)
with col1:
    circles = st.data_editor(circles0, num_rows="dynamic", use_container_width=True, key="circles")
with col2:
    separations = st.data_editor(separations0, num_rows="dynamic", use_container_width=True, key="separations")

col3, col4 = st.columns(2)
with col3:
    field = st.data_editor(field0, num_rows="dynamic", use_container_width=True, key="field")
with col4:
    lab = st.data_editor(lab0, num_rows="dynamic", use_container_width=True, key="lab")

st.subheader("2. Resolver modelo")

if st.button("Ejecutar optimización", type="primary"):
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
        }
        objects = build_model(circles_c, separations_c, field_c, lab_c, params)
        result = solve_lexicographic(objects, time_limit_sec=time_limit_sec)
        summary, field_plan, field_costs, lab_plan = extract_results(objects)

        st.success(f"Estado del solver: {result['status']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Z1 desviación primer nivel", f"{result['z1']:.2f}")
        c2.metric("Z2 desviación interna", f"{result['z2']:.2f}")
        c3.metric("Presupuesto usado", f"${result['objective_budget_used']:,.2f}")

        st.markdown("### Resumen")
        st.dataframe(summary, use_container_width=True)

        st.markdown("### Plan de muestreo de campo")
        st.dataframe(field_plan, use_container_width=True)

        st.markdown("### Costos de campo")
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

with st.expander("Notas del modelo implementado"):
    st.markdown(
        """
        - La cantidad de puntos por círculo y separación se calcula como `ceil(perímetro / separación)`.
        - La etapa 1 minimiza desviaciones entre bloques AHP: campo vs. laboratorio.
        - La etapa 2 minimiza desviaciones internas dentro de campo y laboratorio, fijando el resultado de la etapa 1.
        - La etapa 3 maximiza el presupuesto utilizado, fijando las desviaciones óptimas de las etapas anteriores.
        - Los pesos AHP se normalizan automáticamente si no suman exactamente 1.
        """
    )
