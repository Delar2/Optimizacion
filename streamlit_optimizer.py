import io
import math
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import streamlit as st
import pulp

st.set_page_config(
    page_title="Optimizador MILP-AHP para muestreo de seeps de H₂",
    page_icon="🧪",
    layout="wide",
)

FIELD_ACTIVITIES = {
    "estandar": {
        "label": "Muestreo puntual estándar",
        "time_key": "Tiempo requerido para tomar un muestreo puntual estándar",
        "min_key": "Candidatos de muestreo - Observación directa",
        "ahp_key": "Calidad de un punto de muestreo estándar",
    },
    "multinivel": {
        "label": "Muestreo multinivel",
        "time_key": "Tiempo requerido para hacer un muestreo multinivel",
        "min_key": "Puntos multinivel mínimos por círculo de hadas",
        "ahp_key": "Calidad de un punto multinivel",
    },
    "24h": {
        "label": "Medición de 24 horas",
        "time_key": "Tiempo que consume una medición de 24 horas",
        "min_key": "Mediciones de 24 horas mínimas en toda la estrategia",
        "ahp_key": "Calidad de una medición de 24 horas",
    },
    "extendida": {
        "label": "Medición extendida 5 días",
        "time_key": "Tiempo que consume una medición extendida (5 días)",
        "min_key": "Mediciones extendidas mínimas en toda la estrategia",
        "ahp_key": "Calidad de una medición extendida",
    },
}

LAB_ANALYSES = {
    "cromatografia": {
        "label": "Cromatografía de gas",
        "cost_key": "Análisis de Cromatrografía de Gas",
        "min_key": "Muestras mínimas para cromatografia",
        "ahp_key": "Calidad de cromatografia",
    },
    "deuterio": {
        "label": "Isotopía de deuterio",
        "cost_key": "Análisis de Abundancia Isotópica",
        "min_key": "Muestras mínimas para isotopia de Deuterio",
        "ahp_key": "Calidad de isotopia de Deuterio",
    },
    "helio": {
        "label": "Isotopía de helio",
        "cost_key": "Análisis de Abundancia Isotópica",
        "min_key": "Muestras mínimas para isotopia de Helio",
        "ahp_key": "Calidad de isotopia de Helio",
    },
    "mineralogica": {
        "label": "Caracterización mineralógica",
        "cost_key": "Caracterización Mineralógica",
        "min_key": "Muestras mínimas para caracterizacion mineralogica",
        "ahp_key": "Calidad de caracterizacion mineralogica",
    },
    "biogeoquimica": {
        "label": "Análisis biogeoquímico",
        "cost_key": "Análisis Biogeoquímico",
        "min_key": "Muestras mínimas para biogeoquimica",
        "ahp_key": "Calidad de biogeoquimica",
    },
}

ALIASES = {
    "Presupuesto Total Disponible": "presupuesto_total",
    "Logística de Muestreo (Honorarios, Viáticos y Transporte)": "costo_diario_equipo",
    "Tiempo total disponible para el muestreo": "dias_disponibles",
    "Horas efectivas de trabajo del equipo por día": "horas_dia",
    "Cantidad de equipos/sensores disponibles": "equipos_max",
    "Candidatos de muestreo - Teledetección": "min_circulos_teledeteccion",
}


def norm_text(x: object) -> str:
    s = str(x).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


def fmt_money(x: float) -> str:
    return f"${x:,.0f}".replace(",", ".")


def read_params(file) -> dict:
    raw = pd.read_excel(file, header=None)
    params = {}
    rows = []
    for _, row in raw.iterrows():
        key = row.iloc[0]
        unit = row.iloc[1] if len(row) > 1 else None
        value = row.iloc[2] if len(row) > 2 else None
        if pd.notna(key):
            rows.append({"parametro": str(key).strip(), "unidad": unit, "valor": value})
            if pd.notna(value):
                params[norm_text(key)] = value
    return {"params": params, "table": pd.DataFrame(rows)}


def get_param(params: dict, key: str, default=None, required: bool = True):
    v = params.get(norm_text(key), default)
    if required and v is None:
        raise ValueError(f"No se encontró el parámetro: {key}")
    try:
        return float(v)
    except Exception:
        return v


def read_circles(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    normalized_cols = {norm_text(c): c for c in df.columns}
    id_col = normalized_cols.get("id")
    per_col = normalized_cols.get("perimetro (m)") or normalized_cols.get("perimetro")
    if id_col is None or per_col is None:
        raise ValueError("El archivo de círculos debe tener columnas 'ID' y 'Perimetro (m)'.")
    out = df[[id_col, per_col]].copy()
    out.columns = ["ID", "Perimetro_m"]
    out = out.dropna(subset=["ID", "Perimetro_m"])
    out["ID"] = out["ID"].astype(str)
    out["Perimetro_m"] = pd.to_numeric(out["Perimetro_m"], errors="coerce")
    out = out.dropna(subset=["Perimetro_m"])
    if out.empty:
        raise ValueError("No se encontraron círculos válidos.")
    return out


@dataclass
class ModelData:
    circles: pd.DataFrame
    separations: list[int]
    budget: float
    daily_cost: float
    days: float
    hours_day: float
    max_teams: int
    time_field: dict
    cost_lab: dict
    min_field: dict
    min_lab: dict
    w_field_global: dict
    w_lab_global: dict
    w_block_field: float
    w_block_lab: float


def build_data(circles, params, separations, use_minima=True):
    budget = get_param(params, "Presupuesto Total Disponible")
    daily_cost = get_param(params, "Logística de Muestreo (Honorarios, Viáticos y Transporte)")
    days = get_param(params, "Tiempo total disponible para el muestreo")
    hours_day = get_param(params, "Horas efectivas de trabajo del equipo por día")
    max_teams = int(get_param(params, "Cantidad de equipos/sensores disponibles", 1))

    time_field = {k: get_param(params, v["time_key"]) for k, v in FIELD_ACTIVITIES.items()}
    cost_lab = {k: get_param(params, v["cost_key"]) for k, v in LAB_ANALYSES.items()}

    if use_minima:
        min_field = {k: int(get_param(params, v["min_key"], 0, required=False) or 0) for k, v in FIELD_ACTIVITIES.items()}
        # El mínimo multinivel del Excel está expresado como profundidades por círculo.
        if "multinivel" in min_field:
            min_field["multinivel"] = int(min_field["multinivel"] * len(circles))
        min_lab = {k: int(get_param(params, v["min_key"], 0, required=False) or 0) for k, v in LAB_ANALYSES.items()}
    else:
        min_field = {k: 0 for k in FIELD_ACTIVITIES}
        min_lab = {k: 0 for k in LAB_ANALYSES}

    w_field_global = {k: float(get_param(params, v["ahp_key"])) for k, v in FIELD_ACTIVITIES.items()}
    w_lab_global = {k: float(get_param(params, v["ahp_key"])) for k, v in LAB_ANALYSES.items()}
    total_field = sum(w_field_global.values())
    total_lab = sum(w_lab_global.values())
    total = total_field + total_lab
    w_block_field = total_field / total
    w_block_lab = total_lab / total
    w_field_global = {k: v / total_field for k, v in w_field_global.items()}
    w_lab_global = {k: v / total_lab for k, v in w_lab_global.items()}

    return ModelData(
        circles=circles,
        separations=separations,
        budget=budget,
        daily_cost=daily_cost,
        days=days,
        hours_day=hours_day,
        max_teams=max_teams,
        time_field=time_field,
        cost_lab=cost_lab,
        min_field=min_field,
        min_lab=min_lab,
        w_field_global=w_field_global,
        w_lab_global=w_lab_global,
        w_block_field=w_block_field,
        w_block_lab=w_block_lab,
    )


def solve_model(data: ModelData, limit_lab_by_field=True, solver_time_limit=120):
    I = data.circles["ID"].tolist()
    P = dict(zip(data.circles["ID"], data.circles["Perimetro_m"]))
    S = list(FIELD_ACTIVITIES.keys())
    D = data.separations
    A = list(LAB_ANALYSES.keys())
    npoints = {(i, d): int(math.ceil(P[i] / d)) for i in I for d in D}
    cost_hour = data.daily_cost / data.hours_day

    def create_problem(name="H2_MILP_AHP"):
        prob = pulp.LpProblem(name, pulp.LpMinimize)
        z = pulp.LpVariable.dicts("z", (I, S, D), 0, 1, cat="Binary")
        y = pulp.LpVariable.dicts("y", (I, S), 0, 1, cat="Binary")
        n_lab = pulp.LpVariable.dicts("n_lab", A, lowBound=0, cat="Integer")
        teams = pulp.LpVariable("equipos", lowBound=1, upBound=data.max_teams, cat="Integer")

        field_cost = pulp.LpVariable.dicts("gasto_campo", S, lowBound=0)
        lab_cost = pulp.LpVariable.dicts("gasto_lab", A, lowBound=0)
        q_field = pulp.LpVariable.dicts("cantidad_campo", S, lowBound=0, cat="Integer")
        total_field_cost = pulp.LpVariable("gasto_total_campo", lowBound=0)
        total_lab_cost = pulp.LpVariable("gasto_total_lab", lowBound=0)
        total_used = pulp.LpVariable("presupuesto_utilizado", lowBound=0)

        dp_block_field = pulp.LpVariable("desv_pos_bloque_campo", lowBound=0)
        dn_block_field = pulp.LpVariable("desv_neg_bloque_campo", lowBound=0)
        dp_block_lab = pulp.LpVariable("desv_pos_bloque_lab", lowBound=0)
        dn_block_lab = pulp.LpVariable("desv_neg_bloque_lab", lowBound=0)
        dp_field = pulp.LpVariable.dicts("desv_pos_campo", S, lowBound=0)
        dn_field = pulp.LpVariable.dicts("desv_neg_campo", S, lowBound=0)
        dp_lab = pulp.LpVariable.dicts("desv_pos_lab", A, lowBound=0)
        dn_lab = pulp.LpVariable.dicts("desv_neg_lab", A, lowBound=0)

        for i in I:
            prob += pulp.lpSum(y[i][s] for s in S) >= 1, f"al_menos_una_actividad_{i}"
            for s in S:
                prob += pulp.lpSum(z[i][s][d] for d in D) == y[i][s], f"una_separacion_si_activo_{i}_{s}"

        for s in S:
            prob += q_field[s] == pulp.lpSum(npoints[(i, d)] * z[i][s][d] for i in I for d in D), f"cantidad_{s}"
            prob += field_cost[s] == data.time_field[s] * cost_hour * q_field[s], f"costo_campo_{s}"
            if data.min_field.get(s, 0) > 0:
                prob += q_field[s] >= data.min_field[s], f"minimo_campo_{s}"

        for a in A:
            prob += lab_cost[a] == data.cost_lab[a] * n_lab[a], f"costo_lab_{a}"
            if data.min_lab.get(a, 0) > 0:
                prob += n_lab[a] >= data.min_lab[a], f"minimo_lab_{a}"

        total_field_samples = pulp.lpSum(q_field[s] for s in S)
        if limit_lab_by_field:
            for a in A:
                prob += n_lab[a] <= total_field_samples, f"lab_no_supera_muestras_campo_{a}"

        prob += total_field_cost == pulp.lpSum(field_cost[s] for s in S), "gasto_total_campo_def"
        prob += total_lab_cost == pulp.lpSum(lab_cost[a] for a in A), "gasto_total_lab_def"
        prob += total_used == total_field_cost + total_lab_cost, "presupuesto_usado_def"
        prob += total_used <= data.budget, "presupuesto_total"

        prob += pulp.lpSum(data.time_field[s] * q_field[s] for s in S) <= data.days * data.hours_day * teams, "capacidad_operativa"

        prob += total_field_cost - data.w_block_field * total_used == dp_block_field - dn_block_field, "consistencia_bloque_campo"
        prob += total_lab_cost - data.w_block_lab * total_used == dp_block_lab - dn_block_lab, "consistencia_bloque_lab"

        for s in S:
            prob += field_cost[s] - data.w_field_global[s] * total_field_cost == dp_field[s] - dn_field[s], f"consistencia_campo_{s}"
        for a in A:
            prob += lab_cost[a] - data.w_lab_global[a] * total_lab_cost == dp_lab[a] - dn_lab[a], f"consistencia_lab_{a}"

        Z1 = dp_block_field + dn_block_field + dp_block_lab + dn_block_lab
        Z2 = pulp.lpSum(dp_field[s] + dn_field[s] for s in S) + pulp.lpSum(dp_lab[a] + dn_lab[a] for a in A)
        return prob, locals()

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)

    prob1, v1 = create_problem("H2_MILP_AHP_Z1")
    prob1 += v1["Z1"]
    status1 = prob1.solve(solver)
    if pulp.LpStatus[status1] != "Optimal":
        return {"status": pulp.LpStatus[status1], "message": "No se encontró una solución óptima en la etapa 1."}
    z1_star = pulp.value(v1["Z1"])

    prob2, v2 = create_problem("H2_MILP_AHP_Z2")
    prob2 += v2["Z1"] <= z1_star + 1e-5, "fijar_Z1"
    prob2 += v2["Z2"]
    status2 = prob2.solve(solver)
    if pulp.LpStatus[status2] != "Optimal":
        return {"status": pulp.LpStatus[status2], "message": "No se encontró una solución óptima en la etapa 2."}
    z2_star = pulp.value(v2["Z2"])

    prob3, v3 = create_problem("H2_MILP_AHP_MaxBudget")
    prob3 += v3["Z1"] <= z1_star + 1e-5, "fijar_Z1"
    prob3 += v3["Z2"] <= z2_star + 1e-5, "fijar_Z2"
    prob3 += -v3["total_used"]
    status3 = prob3.solve(solver)
    status = pulp.LpStatus[status3]
    if status != "Optimal":
        return {"status": status, "message": "No se encontró una solución óptima en la etapa 3."}

    z = v3["z"]
    y = v3["y"]
    q_field = v3["q_field"]
    field_cost = v3["field_cost"]
    n_lab = v3["n_lab"]
    lab_cost = v3["lab_cost"]

    assignments = []
    for i in I:
        for s in S:
            if pulp.value(y[i][s]) > 0.5:
                chosen_d = None
                points = 0
                for d in D:
                    if pulp.value(z[i][s][d]) > 0.5:
                        chosen_d = d
                        points = npoints[(i, d)]
                assignments.append({
                    "Círculo": i,
                    "Perímetro (m)": P[i],
                    "Actividad": FIELD_ACTIVITIES[s]["label"],
                    "Separación elegida (m)": chosen_d,
                    "Puntos/muestras": points,
                    "Tiempo unitario (h)": data.time_field[s],
                    "Tiempo total (h)": points * data.time_field[s],
                })

    field_summary = []
    for s in S:
        field_summary.append({
            "Tipo": FIELD_ACTIVITIES[s]["label"],
            "Cantidad": round(pulp.value(q_field[s])),
            "Gasto": pulp.value(field_cost[s]),
            "Peso AHP local": data.w_field_global[s],
            "Mínimo": data.min_field.get(s, 0),
        })

    lab_summary = []
    for a in A:
        lab_summary.append({
            "Tipo": LAB_ANALYSES[a]["label"],
            "Cantidad": round(pulp.value(n_lab[a])),
            "Gasto": pulp.value(lab_cost[a]),
            "Peso AHP local": data.w_lab_global[a],
            "Mínimo": data.min_lab.get(a, 0),
        })

    totals = {
        "Estado": status,
        "Z1_desviación_bloques": z1_star,
        "Z2_desviación_interna": z2_star,
        "Presupuesto total": data.budget,
        "Presupuesto utilizado": pulp.value(v3["total_used"]),
        "Presupuesto no utilizado": data.budget - pulp.value(v3["total_used"]),
        "Gasto campo": pulp.value(v3["total_field_cost"]),
        "Gasto laboratorio": pulp.value(v3["total_lab_cost"]),
        "Equipos seleccionados": round(pulp.value(v3["teams"])),
        "Capacidad disponible (h)": data.days * data.hours_day * round(pulp.value(v3["teams"])),
        "Tiempo de campo utilizado (h)": sum(row["Tiempo total (h)"] for row in assignments),
        "Peso bloque campo": data.w_block_field,
        "Peso bloque laboratorio": data.w_block_lab,
    }
    return {
        "status": status,
        "totals": totals,
        "assignments": pd.DataFrame(assignments),
        "field_summary": pd.DataFrame(field_summary),
        "lab_summary": pd.DataFrame(lab_summary),
    }


def make_excel_download(result):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([result["totals"]]).to_excel(writer, sheet_name="Resumen", index=False)
        result["assignments"].to_excel(writer, sheet_name="Asignacion_campo", index=False)
        result["field_summary"].to_excel(writer, sheet_name="Resumen_campo", index=False)
        result["lab_summary"].to_excel(writer, sheet_name="Resumen_laboratorio", index=False)
        workbook = writer.book
        money_fmt = workbook.add_format({"num_format": "$#,##0"})
        for ws_name in ["Resumen", "Resumen_campo", "Resumen_laboratorio"]:
            ws = writer.sheets[ws_name]
            ws.autofit()
            if ws_name == "Resumen":
                ws.set_column(0, 30, 22)
            else:
                ws.set_column(2, 2, 18, money_fmt)
    return output.getvalue()


st.title("🧪 Optimizador MILP-AHP para muestreo de seeps de hidrógeno")
st.caption("App en Streamlit para aplicar el modelo de Goal Programming jerárquico con consistencia AHP.")

with st.expander("Formato esperado de archivos", expanded=False):
    st.markdown(
        """
        **Circulos.xlsx** debe contener al menos:
        - `ID`
        - `Perimetro (m)`

        **Parametros.xlsx** debe mantener la estructura de tres columnas:
        - Columna A: nombre del parámetro
        - Columna B: unidad
        - Columna C: valor

        La app reconoce los nombres usados en el archivo original: tiempos, costos, mínimos y pesos AHP.
        """
    )

col1, col2 = st.columns(2)
with col1:
    circles_file = st.file_uploader("Sube Circulos.xlsx", type=["xlsx"], key="circles")
with col2:
    params_file = st.file_uploader("Sube Parametros.xlsx", type=["xlsx"], key="params")

st.sidebar.header("Configuración del modelo")
sep_text = st.sidebar.text_input("Separaciones permitidas en metros", value="5, 10, 20, 50, 100")
use_minima = st.sidebar.checkbox("Usar mínimos requeridos del Excel", value=True)
limit_lab = st.sidebar.checkbox("Limitar cada análisis al total de muestras de campo", value=True)
time_limit = st.sidebar.number_input("Tiempo máximo del solver CBC (s)", min_value=10, max_value=600, value=120, step=10)

try:
    separations = [int(float(x.strip())) for x in sep_text.split(",") if x.strip()]
    separations = sorted(set([x for x in separations if x > 0]))
except Exception:
    separations = []

if not circles_file or not params_file:
    st.info("Sube los dos archivos Excel para ejecutar la optimización.")
    st.stop()

if not separations:
    st.error("Define al menos una separación válida, por ejemplo: 5, 10, 20, 50.")
    st.stop()

try:
    circles = read_circles(circles_file)
    parsed = read_params(params_file)
    model_data = build_data(circles, parsed["params"], separations, use_minima=use_minima)
except Exception as e:
    st.error(f"Error leyendo archivos: {e}")
    st.stop()

st.subheader("Datos cargados")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Círculos", len(model_data.circles))
col2.metric("Separaciones", len(model_data.separations))
col3.metric("Presupuesto", fmt_money(model_data.budget))
col4.metric("Equipos máximos", model_data.max_teams)

with st.expander("Ver datos de entrada"):
    st.write("**Círculos**")
    st.dataframe(model_data.circles, use_container_width=True)
    st.write("**Parámetros detectados**")
    st.dataframe(parsed["table"], use_container_width=True)

if st.button("Ejecutar optimización", type="primary"):
    with st.spinner("Resolviendo modelo MILP-AHP en tres etapas lexicográficas..."):
        result = solve_model(model_data, limit_lab_by_field=limit_lab, solver_time_limit=int(time_limit))

    if result.get("status") != "Optimal":
        st.error(result.get("message", "No se encontró solución óptima."))
        st.write("Estado del solver:", result.get("status"))
        st.stop()

    totals = result["totals"]
    st.success("Optimización completada con solución óptima.")

    st.subheader("Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Presupuesto utilizado", fmt_money(totals["Presupuesto utilizado"]))
    c2.metric("Presupuesto no utilizado", fmt_money(totals["Presupuesto no utilizado"]))
    c3.metric("Equipos seleccionados", int(totals["Equipos seleccionados"]))
    c4.metric("Tiempo usado / disponible", f"{totals['Tiempo de campo utilizado (h)']:.0f} / {totals['Capacidad disponible (h)']:.0f} h")

    st.markdown("### Distribución de gasto")
    spend_df = pd.DataFrame([
        {"Bloque": "Campo", "Gasto": totals["Gasto campo"], "Peso AHP bloque": totals["Peso bloque campo"]},
        {"Bloque": "Laboratorio", "Gasto": totals["Gasto laboratorio"], "Peso AHP bloque": totals["Peso bloque laboratorio"]},
    ])
    fig = px.pie(spend_df, names="Bloque", values="Gasto", hole=0.35)
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Asignación por círculo", "Resumen campo", "Resumen laboratorio"])
    with tab1:
        st.dataframe(result["assignments"], use_container_width=True)
    with tab2:
        field_df = result["field_summary"].copy()
        field_df["Gasto"] = field_df["Gasto"].round(0)
        st.dataframe(field_df, use_container_width=True)
        fig_field = px.bar(field_df, x="Tipo", y="Gasto", text="Cantidad")
        st.plotly_chart(fig_field, use_container_width=True)
    with tab3:
        lab_df = result["lab_summary"].copy()
        lab_df["Gasto"] = lab_df["Gasto"].round(0)
        st.dataframe(lab_df, use_container_width=True)
        fig_lab = px.bar(lab_df, x="Tipo", y="Gasto", text="Cantidad")
        st.plotly_chart(fig_lab, use_container_width=True)

    st.download_button(
        "Descargar resultados en Excel",
        data=make_excel_download(result),
        file_name="resultados_optimizacion_h2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
