import io
import math
import re
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pulp
import streamlit as st


st.set_page_config(
    page_title="MILP-AHP Goal Programming",
    page_icon="🧪",
    layout="wide",
)

DEFAULT_SEPARATIONS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

DEFAULT_FIELD_CONFIG = pd.DataFrame(
    {
        "Estrategia de campo": [
            "Muestreo puntual estándar",
            "Muestreo multinivel",
            "Medición 24 horas",
            "Medición extendida",
        ],
        "Tiempo unitario (h)": [1.0, 3.0, 25.0, 121.0],
        "Peso AHP local": [0.2417, 0.2029, 0.1459, 0.0523],
    }
)

DEFAULT_LAB_CONFIG = pd.DataFrame(
    {
        "Análisis de laboratorio": [
            "Mineralógico",
            "Biogeoquímico",
            "Cromatografía",
            "Deuterio",
            "Helio",
        ],
        "Costo unitario": [309_200.0, 3_100_000.0, 1_069_200.0, 792_000.0, 3_240_000.0],
        "Peso AHP local": [0.0399, 0.0328, 0.1511, 0.0897, 0.0438],
    }
)

DEFAULT_PARAMS = {
    "total_days": 60.0,
    "hours_per_day": 8.0,
    "budget": 400_000_000.0,
    "daily_cost": 1_350_000.0,
    "max_teams": 4,
    "separations_text": ", ".join(map(str, DEFAULT_SEPARATIONS)),
    "w_field_block": 0.6428,
    "w_lab_block": 0.3572,
}


# -----------------------------------------------------------------------------
# Utilidades de lectura
# -----------------------------------------------------------------------------

def normalize_text(value) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_number(value, default=None):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    text = text.replace("$", "").replace("COP", "").replace("cop", "").replace(" ", "")
    # 1.234,56 -> 1234.56
    if "," in text and "." in text and text.rfind(",") > text.rfind("."):
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return default


def points_from_perimeter(perimeter: float, separation: float) -> int:
    return max(1, int(math.ceil(float(perimeter) / max(float(separation), 1e-9))))


@st.cache_data
def load_example_circles() -> pd.DataFrame:
    try:
        df = pd.read_excel("Resultados_Optimizacion_Muestreo.xlsx", sheet_name="Estrategia")
        if "Círculo" in df.columns and "Perímetro (m)" in df.columns:
            out = df[["Círculo", "Perímetro (m)"]].copy()
        else:
            raw = pd.read_excel("Resultados_Optimizacion_Muestreo.xlsx", sheet_name="Estrategia", header=None)
            raw.columns = raw.iloc[0]
            out = raw.iloc[1:][["Círculo", "Perímetro (m)"]].copy()
        out.columns = ["Círculo", "Perímetro (m)"]
        out["Perímetro (m)"] = out["Perímetro (m)"].map(lambda x: clean_number(x, np.nan))
        out = out.dropna()
        out = out[out["Perímetro (m)"] > 0]
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(
            {
                "Círculo": [f"C{i}" for i in range(1, 6)],
                "Perímetro (m)": [113, 164, 702, 1393, 221],
            }
        )


def normalize_circles_excel(uploaded_file) -> pd.DataFrame:
    raw = pd.read_excel(uploaded_file)
    raw.columns = [str(c).strip() for c in raw.columns]

    id_candidates = ["ID", "Id", "id", "Círculo", "Circulo", "circle", "Circle"]
    per_candidates = [
        "Perímetro (m)", "Perimetro (m)", "PERIMETRO (m)", "Perímetro", "Perimetro",
        "perimetro", "perímetro", "Perimeter (m)", "Perimeter", "perimeter",
    ]
    id_col = next((c for c in id_candidates if c in raw.columns), raw.columns[0] if len(raw.columns) >= 1 else None)
    per_col = next((c for c in per_candidates if c in raw.columns), raw.columns[1] if len(raw.columns) >= 2 else None)
    if id_col is None or per_col is None:
        raise ValueError("El Excel debe tener columnas ID y Perimetro (m).")

    df = raw[[id_col, per_col]].copy()
    df.columns = ["Círculo", "Perímetro (m)"]
    df["Círculo"] = df["Círculo"].astype(str).str.strip()
    df["Perímetro (m)"] = df["Perímetro (m)"].map(lambda v: clean_number(v, np.nan))
    df = df.dropna(subset=["Círculo", "Perímetro (m)"])
    df = df[df["Perímetro (m)"] > 0]
    df = df.drop_duplicates(subset=["Círculo"], keep="first")
    return df.reset_index(drop=True)


def normalize_weights(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    total = values.sum()
    if total <= 0:
        return pd.Series(np.repeat(1 / len(values), len(values)), index=values.index)
    return values / total


def read_parameters_excel(uploaded_file) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    rows = []
    current_section = ""
    warnings = []

    for _, r in raw.iterrows():
        name = r.iloc[0] if len(r) > 0 else None
        unit = r.iloc[1] if len(r) > 1 else None
        value = r.iloc[2] if len(r) > 2 else None
        n_name = normalize_text(name)
        n_unit = normalize_text(unit)
        n_value = normalize_text(value)
        if not n_name:
            continue
        if n_unit == "unidad" and n_value == "valor":
            current_section = str(name).strip()
            continue
        if "modelo estrategico" in n_name and value is None:
            continue
        rows.append(
            {
                "Sección": current_section,
                "Parámetro": str(name).strip(),
                "Unidad": "" if unit is None else str(unit).strip(),
                "Valor": value,
                "Valor numérico": clean_number(value, np.nan),
                "Clave": n_name,
            }
        )

    params_table = pd.DataFrame(rows)
    params = DEFAULT_PARAMS.copy()
    field_df = DEFAULT_FIELD_CONFIG.copy()
    lab_df = DEFAULT_LAB_CONFIG.copy()

    def row_value(all_tokens, any_tokens=None, exclude=None, default=None):
        any_tokens = any_tokens or []
        exclude = exclude or []
        for _, rr in params_table.iterrows():
            key = rr["Clave"]
            if all(t in key for t in all_tokens):
                if any_tokens and not any(t in key for t in any_tokens):
                    continue
                if any(t in key for t in exclude):
                    continue
                val = clean_number(rr["Valor"], default)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return val
        return default

    def row_text(all_tokens, any_tokens=None, exclude=None, default=None):
        any_tokens = any_tokens or []
        exclude = exclude or []
        for _, rr in params_table.iterrows():
            key = rr["Clave"]
            if all(t in key for t in all_tokens):
                if any_tokens and not any(t in key for t in any_tokens):
                    continue
                if any(t in key for t in exclude):
                    continue
                val = rr["Valor"]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return str(val).strip()
        return default

    # Parámetros generales
    params["total_days"] = row_value(["tiempo", "total", "disponible"], default=params["total_days"])
    params["hours_per_day"] = row_value(["horas", "efectivas"], default=params["hours_per_day"])
    params["budget"] = row_value(["presupuesto", "total"], default=params["budget"])
    params["daily_cost"] = row_value(["costo", "diario", "muestreo"], default=params["daily_cost"])
    params["max_teams"] = int(row_value(["equipos"], any_tokens=["disponibles", "sensores"], default=params["max_teams"]))

    sep_text = row_text(["separaciones"], any_tokens=["permitidas", "puntos"], default=params["separations_text"])
    if sep_text:
        params["separations_text"] = sep_text

    # Tiempos de estrategias de campo. Compatible con el Excel anterior.
    field_map = {
        "Muestreo puntual estándar": [["muestreo", "puntual", "estandar"], ["punto", "estandar"]],
        "Muestreo multinivel": [["muestreo", "multinivel"], ["punto", "multinivel"]],
        "Medición 24 horas": [["medicion", "24"], ["medicion", "24"]],
        "Medición extendida": [["medicion", "extendida"], ["medicion", "extendida"]],
    }
    for name, (time_tokens, weight_tokens) in field_map.items():
        idx = field_df.index[field_df["Estrategia de campo"] == name]
        if len(idx) == 0:
            continue
        i = idx[0]
        field_df.at[i, "Tiempo unitario (h)"] = row_value(time_tokens, default=field_df.at[i, "Tiempo unitario (h)"])
        # Acepta filas tipo "Calidad de..." o "Peso AHP local...".
        field_df.at[i, "Peso AHP local"] = row_value(["calidad"] + weight_tokens, default=field_df.at[i, "Peso AHP local"])
        field_df.at[i, "Peso AHP local"] = row_value(["peso", "ahp"] + weight_tokens, default=field_df.at[i, "Peso AHP local"])

    lab_map = {
        "Mineralógico": "mineralogica",
        "Biogeoquímico": "biogeoquimica",
        "Cromatografía": "cromatografia",
        "Deuterio": "deuterio",
        "Helio": "helio",
    }
    for name, token in lab_map.items():
        idx = lab_df.index[lab_df["Análisis de laboratorio"] == name]
        if len(idx) == 0:
            continue
        i = idx[0]
        lab_df.at[i, "Costo unitario"] = row_value(["costo", "unitario", token], default=lab_df.at[i, "Costo unitario"])
        lab_df.at[i, "Peso AHP local"] = row_value(["calidad", token], default=lab_df.at[i, "Peso AHP local"])
        lab_df.at[i, "Peso AHP local"] = row_value(["peso", "ahp", token], default=lab_df.at[i, "Peso AHP local"])

    explicit_w_field = row_value(["peso", "ahp", "campo"], default=None)
    explicit_w_lab = row_value(["peso", "ahp", "laboratorio"], default=None)
    if explicit_w_field is not None and explicit_w_lab is not None and explicit_w_field + explicit_w_lab > 0:
        total = explicit_w_field + explicit_w_lab
        params["w_field_block"] = explicit_w_field / total
        params["w_lab_block"] = explicit_w_lab / total
    else:
        # Si el Excel antiguo solo trae pesos globales de calidad, se derivan proporciones de bloque.
        field_sum = float(pd.to_numeric(field_df["Peso AHP local"], errors="coerce").fillna(0).sum())
        lab_sum = float(pd.to_numeric(lab_df["Peso AHP local"], errors="coerce").fillna(0).sum())
        if field_sum + lab_sum > 0:
            params["w_field_block"] = field_sum / (field_sum + lab_sum)
            params["w_lab_block"] = lab_sum / (field_sum + lab_sum)
            warnings.append(
                "No encontré pesos AHP explícitos para los bloques Campo/Laboratorio; los derivé a partir de las sumas de pesos cargados."
            )

    params_table = params_table.drop(columns=["Clave"])
    return params, params_table, field_df, lab_df, warnings


# -----------------------------------------------------------------------------
# Modelo MILP-AHP Goal Programming lexicográfico
# -----------------------------------------------------------------------------

def solve_milp_ahp_goal_programming(
    circles_df: pd.DataFrame,
    separations: List[int],
    field_config: pd.DataFrame,
    lab_config: pd.DataFrame,
    total_days: float,
    hours_per_day: float,
    budget: float,
    daily_cost: float,
    max_teams: int,
    w_field_block: float,
    w_lab_block: float,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    circles = circles_df["Círculo"].astype(str).tolist()
    perimeter = dict(zip(circles, circles_df["Perímetro (m)"].astype(float)))
    field_types = field_config["Estrategia de campo"].astype(str).tolist()
    lab_types = lab_config["Análisis de laboratorio"].astype(str).tolist()

    t_field = dict(zip(field_types, field_config["Tiempo unitario (h)"].astype(float)))
    w_field_local = dict(zip(field_types, normalize_weights(field_config["Peso AHP local"])))
    c_lab = dict(zip(lab_types, lab_config["Costo unitario"].astype(float)))
    w_lab_local = dict(zip(lab_types, normalize_weights(lab_config["Peso AHP local"])))

    # Normalización de pesos de primer nivel.
    w_sum = float(w_field_block + w_lab_block)
    if w_sum <= 0:
        w_field_block, w_lab_block = 0.5, 0.5
    else:
        w_field_block, w_lab_block = w_field_block / w_sum, w_lab_block / w_sum

    n_points = {(i, s): points_from_perimeter(perimeter[i], s) for i in circles for s in separations}
    cost_per_field_hour = daily_cost / max(hours_per_day, 1e-9)

    prob = pulp.LpProblem("MILP_AHP_Goal_Programming", pulp.LpMinimize)

    z = pulp.LpVariable.dicts("activar", (circles, field_types), 0, 1, cat="Binary")
    x = pulp.LpVariable.dicts("usar_separacion", (circles, field_types, separations), 0, 1, cat="Binary")
    lab = pulp.LpVariable.dicts("muestras_lab", lab_types, lowBound=0, cat="Integer")
    teams = pulp.LpVariable("equipos", lowBound=1, upBound=max_teams, cat="Integer")

    field_spend = pulp.LpVariable.dicts("gasto_campo", field_types, lowBound=0, cat="Continuous")
    lab_spend = pulp.LpVariable.dicts("gasto_lab", lab_types, lowBound=0, cat="Continuous")
    total_used = pulp.LpVariable("presupuesto_usado", lowBound=0, upBound=budget, cat="Continuous")
    total_field = pulp.LpVariable("gasto_total_campo", lowBound=0, cat="Continuous")
    total_lab = pulp.LpVariable("gasto_total_lab", lowBound=0, cat="Continuous")

    d_block_field = pulp.LpVariable("desv_bloque_campo", lowBound=0, cat="Continuous")
    d_block_lab = pulp.LpVariable("desv_bloque_laboratorio", lowBound=0, cat="Continuous")
    d_field = pulp.LpVariable.dicts("desv_interna_campo", field_types, lowBound=0, cat="Continuous")
    d_lab = pulp.LpVariable.dicts("desv_interna_lab", lab_types, lowBound=0, cat="Continuous")

    quantity_field = {}
    for f in field_types:
        quantity_field[f] = pulp.lpSum(n_points[(i, s)] * x[i][f][s] for i in circles for s in separations)

    # 5.1 Activación: si se activa una actividad en un círculo, elige exactamente una separación.
    for i in circles:
        for f in field_types:
            prob += pulp.lpSum(x[i][f][s] for s in separations) == z[i][f], f"activacion_{i}_{f}"

    # 5.2 Integralidad: cada círculo debe tener al menos una actividad activa.
    for i in circles:
        prob += pulp.lpSum(z[i][f] for f in field_types) >= 1, f"circulo_con_actividad_{i}"

    # 5.4 Costos de campo.
    for f in field_types:
        prob += field_spend[f] == cost_per_field_hour * t_field[f] * quantity_field[f], f"costo_campo_{f}"

    # 5.5 Costos de laboratorio.
    for k in lab_types:
        prob += lab_spend[k] == c_lab[k] * lab[k], f"costo_lab_{k}"

    # 5.6 Gastos agregados.
    prob += total_field == pulp.lpSum(field_spend[f] for f in field_types), "gasto_total_campo"
    prob += total_lab == pulp.lpSum(lab_spend[k] for k in lab_types), "gasto_total_laboratorio"
    prob += total_used == total_field + total_lab, "presupuesto_usado_def"

    # 5.7 Presupuesto total.
    prob += total_used <= budget, "presupuesto_total"

    # 5.8 Capacidad operativa.
    total_field_hours = pulp.lpSum(t_field[f] * quantity_field[f] for f in field_types)
    prob += total_field_hours <= total_days * hours_per_day * teams, "capacidad_operativa"

    # 5.10 Consistencia AHP de primer nivel.
    prob += d_block_field >= total_field - w_field_block * total_used, "desv_bloque_campo_pos"
    prob += d_block_field >= -(total_field - w_field_block * total_used), "desv_bloque_campo_neg"
    prob += d_block_lab >= total_lab - w_lab_block * total_used, "desv_bloque_lab_pos"
    prob += d_block_lab >= -(total_lab - w_lab_block * total_used), "desv_bloque_lab_neg"

    # 5.11 Consistencia AHP dentro del bloque de campo.
    for f in field_types:
        prob += d_field[f] >= field_spend[f] - w_field_local[f] * total_field, f"desv_campo_pos_{f}"
        prob += d_field[f] >= -(field_spend[f] - w_field_local[f] * total_field), f"desv_campo_neg_{f}"

    # 5.12 Consistencia AHP dentro del bloque de laboratorio.
    for k in lab_types:
        prob += d_lab[k] >= lab_spend[k] - w_lab_local[k] * total_lab, f"desv_lab_pos_{k}"
        prob += d_lab[k] >= -(lab_spend[k] - w_lab_local[k] * total_lab), f"desv_lab_neg_{k}"

    z1_expr = d_block_field + d_block_lab
    z2_expr = pulp.lpSum(d_field[f] for f in field_types) + pulp.lpSum(d_lab[k] for k in lab_types)

    solver = pulp.PULP_CBC_CMD(msg=False)
    tolerance = max(1.0, budget * 1e-7)

    # Etapa 1: minimizar desviaciones de primer nivel.
    prob.sense = pulp.LpMinimize
    prob.setObjective(z1_expr)
    prob.solve(solver)
    status1 = pulp.LpStatus[prob.status]
    if status1 not in ["Optimal", "Feasible"]:
        return status1, pd.DataFrame(), pd.DataFrame(), {}
    z1_star = float(pulp.value(z1_expr) or 0)
    prob += z1_expr <= z1_star + tolerance, "fijar_Z1"

    # Etapa 2: minimizar desviaciones internas.
    prob.setObjective(z2_expr)
    prob.solve(solver)
    status2 = pulp.LpStatus[prob.status]
    if status2 not in ["Optimal", "Feasible"]:
        return status2, pd.DataFrame(), pd.DataFrame(), {}
    z2_star = float(pulp.value(z2_expr) or 0)
    prob += z2_expr <= z2_star + tolerance, "fijar_Z2"

    # Etapa 3: maximizar presupuesto utilizado conservando Z1* y Z2*.
    prob.sense = pulp.LpMaximize
    prob.setObjective(total_used)
    prob.solve(solver)
    status3 = pulp.LpStatus[prob.status]
    if status3 not in ["Optimal", "Feasible"]:
        return status3, pd.DataFrame(), pd.DataFrame(), {}

    # Salidas.
    strategy_rows = []
    for i in circles:
        for f in field_types:
            if (pulp.value(z[i][f]) or 0) > 0.5:
                selected_sep = None
                for s in separations:
                    if (pulp.value(x[i][f][s]) or 0) > 0.5:
                        selected_sep = s
                        break
                if selected_sep is None:
                    continue
                pts = n_points[(i, selected_sep)]
                hours = pts * t_field[f]
                cost = hours * cost_per_field_hour
                strategy_rows.append(
                    {
                        "Círculo": i,
                        "Perímetro (m)": perimeter[i],
                        "Estrategia de campo": f,
                        "Separación seleccionada (m)": selected_sep,
                        "Puntos / mediciones": pts,
                        "Tiempo requerido (h)": hours,
                        "Gasto campo": cost,
                    }
                )
    strategy_df = pd.DataFrame(strategy_rows)

    lab_rows = []
    for k in lab_types:
        samples = int(round(pulp.value(lab[k]) or 0))
        lab_rows.append(
            {
                "Análisis de laboratorio": k,
                "Muestras": samples,
                "Costo unitario": c_lab[k],
                "Gasto laboratorio": samples * c_lab[k],
            }
        )
    lab_df = pd.DataFrame(lab_rows)

    total_field_value = float(pulp.value(total_field) or 0)
    total_lab_value = float(pulp.value(total_lab) or 0)
    total_used_value = float(pulp.value(total_used) or 0)
    total_hours_value = float(pulp.value(total_field_hours) or 0)
    teams_value = int(round(pulp.value(teams) or 1))

    metrics = {
        "Estado etapa 1": status1,
        "Estado etapa 2": status2,
        "Estado final": status3,
        "Z1 desviación primer nivel": z1_star,
        "Z2 desviación interna": z2_star,
        "Presupuesto usado": total_used_value,
        "Presupuesto disponible": float(budget),
        "Margen presupuesto": float(budget - total_used_value),
        "Gasto campo": total_field_value,
        "Gasto laboratorio": total_lab_value,
        "Proporción campo real": total_field_value / total_used_value if total_used_value else 0,
        "Proporción laboratorio real": total_lab_value / total_used_value if total_used_value else 0,
        "Proporción campo objetivo AHP": float(w_field_block),
        "Proporción laboratorio objetivo AHP": float(w_lab_block),
        "Horas campo requeridas": total_hours_value,
        "Días-equipo requeridos": total_hours_value / max(hours_per_day, 1e-9),
        "Días calendario estimados": total_hours_value / max(hours_per_day * teams_value, 1e-9),
        "Equipos seleccionados": teams_value,
        "Número de círculos": len(circles),
        "Actividades de campo activas": len(strategy_df),
    }

    return status3, strategy_df, lab_df, metrics


def to_excel(strategy_df: pd.DataFrame, lab_df: pd.DataFrame, metrics: Dict[str, float], params_table: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        strategy_df.to_excel(writer, index=False, sheet_name="Campo")
        lab_df.to_excel(writer, index=False, sheet_name="Laboratorio")
        pd.DataFrame(list(metrics.items()), columns=["Concepto", "Valor"]).to_excel(writer, index=False, sheet_name="Resumen")
        if params_table is not None and not params_table.empty:
            params_table.to_excel(writer, index=False, sheet_name="Parametros cargados")

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        money_fmt = workbook.add_format({"num_format": "$#,##0"})
        num_fmt = workbook.add_format({"num_format": "0.00"})
        for sheet_name, ws in writer.sheets.items():
            ws.set_row(0, None, header_fmt)
            ws.set_column(0, 0, 26)
            ws.set_column(1, 10, 22)
        if "Campo" in writer.sheets:
            writer.sheets["Campo"].set_column(6, 6, 18, money_fmt)
        if "Laboratorio" in writer.sheets:
            writer.sheets["Laboratorio"].set_column(2, 3, 18, money_fmt)
        if "Resumen" in writer.sheets:
            writer.sheets["Resumen"].set_column(1, 1, 24, num_fmt)
    return output.getvalue()


# -----------------------------------------------------------------------------
# Interfaz Streamlit
# -----------------------------------------------------------------------------
st.title("🧪 MILP-AHP Goal Programming para estrategias de muestreo")
st.caption(
    "Asignación de recursos de campo y laboratorio según consistencia con prioridades AHP, "
    "presupuesto, tiempo, geometría y capacidad operativa."
)

with st.sidebar:
    st.header("1. Archivos de entrada")
    uploaded_params_file = st.file_uploader(
        "Excel de parámetros",
        type=["xlsx", "xls"],
        help="Formato esperado: columna A = Parámetro, columna B = Unidad, columna C = Valor.",
    )
    uploaded_circles_file = st.file_uploader(
        "Excel de círculos de hadas",
        type=["xlsx", "xls"],
        help="Formato esperado: columnas ID y Perimetro (m).",
    )
    use_example = st.checkbox("Usar datos de ejemplo si no subo círculos", value=True)
    n_circles = st.number_input("Número de círculos ejemplo/manual", min_value=1, max_value=500, value=30, step=1)
    allow_edit_circles = st.checkbox("Permitir editar círculos", value=False)

params = DEFAULT_PARAMS.copy()
params_table = pd.DataFrame()
field_config_base = DEFAULT_FIELD_CONFIG.copy()
lab_config_base = DEFAULT_LAB_CONFIG.copy()
param_warnings = []
param_source = "Valores por defecto"

if uploaded_params_file is not None:
    try:
        params, params_table, field_config_base, lab_config_base, param_warnings = read_parameters_excel(uploaded_params_file)
        param_source = "Excel de parámetros subido"
    except Exception as exc:
        st.error(f"No pude leer el Excel de parámetros: {exc}")
        st.info("Usa el formato: Parámetro | Unidad | Valor.")

st.subheader("A. Parámetros cargados")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Fuente", param_source)
col_b.metric("Parámetros leídos", len(params_table) if not params_table.empty else 0)
col_c.metric("Modelo", "MILP-AHP Goal Programming")
if not params_table.empty:
    st.dataframe(params_table[["Sección", "Parámetro", "Unidad", "Valor"]], use_container_width=True, hide_index=True)
else:
    st.info("Puedes subir un Excel de parámetros o usar los valores por defecto.")
for msg in param_warnings:
    st.warning(msg)

st.subheader("B. Geometría de círculos")
try:
    if uploaded_circles_file is not None:
        base_circles = normalize_circles_excel(uploaded_circles_file)
        circle_source = "Excel de círculos subido"
    elif use_example:
        base_circles = load_example_circles().head(int(n_circles)).copy()
        circle_source = "Datos de ejemplo"
    else:
        base_circles = pd.DataFrame(
            {"Círculo": [f"C{i}" for i in range(1, int(n_circles) + 1)], "Perímetro (m)": [100.0] * int(n_circles)}
        )
        circle_source = "Tabla manual"
except Exception as exc:
    st.error(f"No pude leer el Excel de círculos: {exc}")
    base_circles = pd.DataFrame(columns=["Círculo", "Perímetro (m)"])
    circle_source = "Error de lectura"

m1, m2, m3 = st.columns(3)
m1.metric("Número de círculos", len(base_circles))
m2.metric("Perímetro total (m)", f"{base_circles['Perímetro (m)'].sum():,.1f}" if not base_circles.empty else "0")
m3.metric("Fuente", circle_source)

if allow_edit_circles:
    circles_df = st.data_editor(
        base_circles,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"Perímetro (m)": st.column_config.NumberColumn(min_value=0.01, step=1.0)},
    )
else:
    circles_df = base_circles.copy()
    st.dataframe(circles_df, use_container_width=True, hide_index=True)

st.subheader("C. Restricciones generales")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    total_days = st.number_input("Días disponibles", min_value=1.0, value=float(params["total_days"]), step=1.0)
with c2:
    hours_per_day = st.number_input("Horas efectivas/día", min_value=1.0, value=float(params["hours_per_day"]), step=1.0)
with c3:
    max_teams = st.number_input("Equipos máximos", min_value=1, max_value=100, value=int(params["max_teams"]), step=1)
with c4:
    budget = st.number_input("Presupuesto total", min_value=1.0, value=float(params["budget"]), step=1_000_000.0)
with c5:
    daily_cost = st.number_input("Costo diario equipo", min_value=0.0, value=float(params["daily_cost"]), step=50_000.0)

sep_text = st.text_input("Separaciones permitidas entre puntos (m)", value=str(params["separations_text"]))
try:
    separations = sorted({int(float(x.strip().replace(',', '.'))) for x in str(sep_text).split(',') if x.strip()})
except Exception:
    separations = DEFAULT_SEPARATIONS
    st.warning("No pude interpretar las separaciones. Se usarán las separaciones por defecto.")

st.subheader("D. Pesos AHP de primer nivel")
w1, w2 = st.columns(2)
with w1:
    w_field_block = st.number_input("Peso AHP bloque Campo", min_value=0.0, max_value=1.0, value=float(params["w_field_block"]), step=0.01, format="%.4f")
with w2:
    w_lab_block = st.number_input("Peso AHP bloque Laboratorio", min_value=0.0, max_value=1.0, value=float(params["w_lab_block"]), step=0.01, format="%.4f")
if w_field_block + w_lab_block <= 0:
    st.error("Los pesos de Campo y Laboratorio no pueden sumar cero.")
else:
    st.caption(f"Pesos normalizados usados: Campo = {w_field_block/(w_field_block+w_lab_block):.4f}, Laboratorio = {w_lab_block/(w_field_block+w_lab_block):.4f}")

st.subheader("E. Estrategias de muestreo de campo")
st.caption("Cada estrategia puede activarse o no en cada círculo. Si se activa, el modelo elige una separación permitida.")
field_config = st.data_editor(
    field_config_base,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Tiempo unitario (h)": st.column_config.NumberColumn(min_value=0.0, step=0.25),
        "Peso AHP local": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"),
    },
)
field_config = field_config.dropna(subset=["Estrategia de campo"])
field_config["Peso AHP local normalizado"] = normalize_weights(field_config["Peso AHP local"]).values
st.dataframe(field_config[["Estrategia de campo", "Peso AHP local", "Peso AHP local normalizado"]], use_container_width=True, hide_index=True)

st.subheader("F. Análisis de laboratorio")
st.caption("El modelo decide el número de muestras de laboratorio de cada tipo para aproximar las proporciones AHP locales.")
lab_config = st.data_editor(
    lab_config_base,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Costo unitario": st.column_config.NumberColumn(min_value=0.0, step=10_000.0),
        "Peso AHP local": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"),
    },
)
lab_config = lab_config.dropna(subset=["Análisis de laboratorio"])
lab_config["Peso AHP local normalizado"] = normalize_weights(lab_config["Peso AHP local"]).values
st.dataframe(lab_config[["Análisis de laboratorio", "Peso AHP local", "Peso AHP local normalizado"]], use_container_width=True, hide_index=True)

run = st.button("Resolver modelo MILP-AHP", type="primary", use_container_width=True)

if run:
    if circles_df.empty:
        st.error("No hay círculos de hadas para optimizar.")
    elif not separations:
        st.error("No hay separaciones permitidas.")
    elif field_config.empty:
        st.error("No hay estrategias de campo.")
    elif lab_config.empty:
        st.error("No hay análisis de laboratorio.")
    elif w_field_block + w_lab_block <= 0:
        st.error("Corrige los pesos AHP de primer nivel.")
    else:
        with st.spinner("Resolviendo Goal Programming lexicográfico en tres etapas..."):
            status, strategy_df, lab_df, metrics = solve_milp_ahp_goal_programming(
                circles_df=circles_df,
                separations=separations,
                field_config=field_config,
                lab_config=lab_config,
                total_days=float(total_days),
                hours_per_day=float(hours_per_day),
                budget=float(budget),
                daily_cost=float(daily_cost),
                max_teams=int(max_teams),
                w_field_block=float(w_field_block),
                w_lab_block=float(w_lab_block),
            )

        if status not in ["Optimal", "Feasible"]:
            st.error(f"No se encontró solución factible. Estado: {status}")
            st.info("Prueba aumentar presupuesto/tiempo, reducir actividades de campo o aumentar equipos disponibles.")
        else:
            st.success(f"Solución encontrada: {status}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Presupuesto usado", f"${metrics['Presupuesto usado']:,.0f}")
            k2.metric("Gasto campo", f"${metrics['Gasto campo']:,.0f}")
            k3.metric("Gasto laboratorio", f"${metrics['Gasto laboratorio']:,.0f}")
            k4.metric("Días calendario", f"{metrics['Días calendario estimados']:.1f}")

            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Z1 primer nivel", f"{metrics['Z1 desviación primer nivel']:,.2f}")
            k6.metric("Z2 interno", f"{metrics['Z2 desviación interna']:,.2f}")
            k7.metric("Equipos", int(metrics["Equipos seleccionados"]))
            k8.metric("Actividades activas", int(metrics["Actividades de campo activas"]))

            st.markdown("### Resultado de campo")
            st.dataframe(strategy_df, use_container_width=True, hide_index=True)

            st.markdown("### Resultado de laboratorio")
            st.dataframe(lab_df, use_container_width=True, hide_index=True)

            st.markdown("### Resumen del modelo")
            st.json(metrics)

            excel_bytes = to_excel(strategy_df, lab_df, metrics, params_table)
            st.download_button(
                "Descargar resultados en Excel",
                data=excel_bytes,
                file_name="resultados_milp_ahp_goal_programming.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
