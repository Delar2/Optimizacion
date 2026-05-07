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
    page_title="Optimizador de muestreo MILP",
    page_icon="🧪",
    layout="wide",
)


DEFAULT_SEPARATIONS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
DEFAULT_LAB_TYPES = [
    "Mineralógico",
    "Biogeoquímico",
    "Cromatografía",
    "Deuterio",
    "Helio",
]

DEFAULT_PARAMS = {
    "total_days": 60.0,
    "hours_per_day": 8.0,
    "t_standard": 1.0,
    "t_multilevel": 3.0,
    "t_24h": 25.0,
    "t_extended": 121.0,
    "budget": 400_000_000.0,
    "daily_cost": 1_350_000.0,
    "min_multilevel_per_circle": 3,
    "min_24h": 10,
    "min_extended": 2,
    "max_teams": 4,
    "w_standard": 0.2417,
    "w_multilevel": 0.2029,
    "w_24h": 0.1459,
    "w_extended": 0.0523,
    # Parámetros fijos de optimización que también pueden venir desde Excel.
    "model_choice": "Comparar MILP mejorado vs genético",
    "multilevel_rule": "Según perímetro",
    "multilevel_pct": 0.10,
    "ga_generations": 120,
    "ga_population": 80,
    "separations_text": ", ".join(map(str, DEFAULT_SEPARATIONS)),
}

DEFAULT_LAB_CONFIG = pd.DataFrame(
    {
        "Tipo de análisis": DEFAULT_LAB_TYPES,
        "Mínimo de muestras": [15, 15, 15, 5, 5],
        "Costo unitario": [309_200, 3_100_000, 1_069_200, 792_000, 3_240_000],
        "Peso calidad": [0.0399, 0.0328, 0.1511, 0.0897, 0.0438],
    }
)


# -----------------------------
# Lectura y normalización
# -----------------------------

def normalize_text(value) -> str:
    """Normaliza texto para comparar etiquetas con/sin acentos."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_number(value, default=None):
    """Convierte números desde Excel aceptando coma o punto decimal."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    text = text.replace("$", "").replace("COP", "").replace(" ", "")
    # Si hay punto de miles y coma decimal: 1.234,56 -> 1234.56
    if "," in text and "." in text and text.rfind(",") > text.rfind("."):
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return default


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


def normalize_circles_excel(uploaded_file) -> pd.DataFrame:
    """Lee un Excel con columnas tipo ID y Perimetro (m), normaliza nombres y valida datos."""
    raw = pd.read_excel(uploaded_file)
    raw.columns = [str(c).strip() for c in raw.columns]

    id_candidates = [
        "ID", "Id", "id", "Círculo", "Circulo", "circulo", "CIRCLE", "Circle", "circle"
    ]
    perimeter_candidates = [
        "Perímetro (m)", "Perimetro (m)", "PERIMETRO (m)", "Perímetro", "Perimetro",
        "perimetro", "perímetro", "Perimeter (m)", "Perimeter", "perimeter"
    ]

    id_col = next((c for c in id_candidates if c in raw.columns), None)
    perimeter_col = next((c for c in perimeter_candidates if c in raw.columns), None)

    if id_col is None and len(raw.columns) >= 1:
        id_col = raw.columns[0]
    if perimeter_col is None and len(raw.columns) >= 2:
        perimeter_col = raw.columns[1]

    if id_col is None or perimeter_col is None:
        raise ValueError("El Excel debe tener una columna de ID y una columna de perímetro en metros.")

    df = raw[[id_col, perimeter_col]].copy()
    df.columns = ["Círculo", "Perímetro (m)"]
    df["Círculo"] = df["Círculo"].astype(str).str.strip()
    df["Perímetro (m)"] = df["Perímetro (m)"].map(lambda v: clean_number(v, np.nan))
    df = df.dropna(subset=["Círculo", "Perímetro (m)"])
    df = df[df["Perímetro (m)"] > 0]
    df = df.drop_duplicates(subset=["Círculo"], keep="first")
    return df.reset_index(drop=True)


def read_strategy_parameters_excel(uploaded_file) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Lee el Excel de parámetros con formato:
    Columna A = parámetro/sección, Columna B = unidad, Columna C = valor.
    Devuelve parámetros generales, tabla normalizada, configuración de laboratorio y advertencias.
    """
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

        # Filas de sección: tienen cabecera tipo "PARAMETROS DE TIEMPO | Unidad | Valor".
        if n_unit == "unidad" and n_value == "valor":
            current_section = str(name).strip()
            continue

        # Título superior del archivo.
        if "modelo estrategico" in n_name and value is None:
            continue

        rows.append(
            {
                "Sección": current_section,
                "Parámetro": str(name).strip(),
                "Unidad": "" if unit is None else str(unit).strip(),
                "Valor": value,
                "Valor numérico": clean_number(value, np.nan),
                "Clave normalizada": n_name,
            }
        )

    params_table = pd.DataFrame(rows)
    params = DEFAULT_PARAMS.copy()
    lab_df = DEFAULT_LAB_CONFIG.copy()

    def row_value(contains_all: List[str], contains_any: List[str] = None, exclude: List[str] = None, default=None):
        contains_any = contains_any or []
        exclude = exclude or []
        for _, rr in params_table.iterrows():
            key = rr["Clave normalizada"]
            if all(token in key for token in contains_all):
                if contains_any and not any(token in key for token in contains_any):
                    continue
                if any(token in key for token in exclude):
                    continue
                val = clean_number(rr["Valor"], default)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return val
        return default

    def row_text(contains_all: List[str], contains_any: List[str] = None, exclude: List[str] = None, default=None):
        contains_any = contains_any or []
        exclude = exclude or []
        for _, rr in params_table.iterrows():
            key = rr["Clave normalizada"]
            if all(token in key for token in contains_all):
                if contains_any and not any(token in key for token in contains_any):
                    continue
                if any(token in key for token in exclude):
                    continue
                value = rr["Valor"]
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    return str(value).strip()
        return default

    # Tiempo
    params["total_days"] = row_value(["tiempo", "total", "disponible"], default=params["total_days"])
    params["hours_per_day"] = row_value(["horas", "efectivas"], default=params["hours_per_day"])
    params["t_standard"] = row_value(["muestreo", "puntual", "estandar"], default=params["t_standard"])
    params["t_multilevel"] = row_value(["muestreo", "multinivel"], default=params["t_multilevel"])
    params["t_24h"] = row_value(["medicion", "24"], default=params["t_24h"])
    params["t_extended"] = row_value(["medicion", "extendida"], default=params["t_extended"])

    # Costos
    params["budget"] = row_value(["presupuesto", "total"], default=params["budget"])
    params["daily_cost"] = row_value(["costo", "diario", "muestreo"], default=params["daily_cost"])

    # Mínimos y equipos
    params["min_multilevel_per_circle"] = int(row_value(["puntos", "multinivel", "minimos"], default=params["min_multilevel_per_circle"]))
    params["min_24h"] = int(row_value(["mediciones", "24", "minimas"], default=params["min_24h"]))
    params["min_extended"] = int(row_value(["mediciones", "extendidas", "minimas"], default=params["min_extended"]))
    params["max_teams"] = int(row_value(["equipos", "sensores", "disponibles"], default=params["max_teams"]))

    # Pesos principales
    params["w_standard"] = row_value(["calidad", "punto", "estandar"], default=params["w_standard"])
    params["w_multilevel"] = row_value(["calidad", "punto", "multinivel"], default=params["w_multilevel"])
    params["w_24h"] = row_value(["calidad", "medicion", "24"], default=params["w_24h"])
    params["w_extended"] = row_value(["calidad", "medicion", "extendida"], default=params["w_extended"])

    lab_mapping = {
        "Cromatografía": ["cromatografia"],
        "Deuterio": ["deuterio"],
        "Helio": ["helio"],
        "Mineralógico": ["mineralogica"],
        "Biogeoquímico": ["biogeoquimica"],
    }

    for lab_name, tokens in lab_mapping.items():
        idx = lab_df.index[lab_df["Tipo de análisis"] == lab_name]
        if len(idx) == 0:
            continue
        i = idx[0]
        token = tokens[0]
        cost = row_value(["costo", "unitario", token], default=lab_df.at[i, "Costo unitario"])
        minimum = row_value(["muestras", "minimas", token], default=lab_df.at[i, "Mínimo de muestras"])
        weight = row_value(["calidad", token], default=lab_df.at[i, "Peso calidad"])
        lab_df.at[i, "Costo unitario"] = float(cost)
        lab_df.at[i, "Mínimo de muestras"] = int(minimum)
        lab_df.at[i, "Peso calidad"] = float(weight)

    # Parámetros de configuración del optimizador.
    model_text = row_text(["modelo", "optimizacion"], default=params["model_choice"])
    model_key = normalize_text(model_text)
    if "actual" in model_key:
        params["model_choice"] = "MILP actual"
    elif "genetico" in model_key and "compar" in model_key:
        params["model_choice"] = "Comparar MILP mejorado vs genético"
    elif "genetico" in model_key:
        params["model_choice"] = "Algoritmo genético"
    elif "mejorado" in model_key:
        params["model_choice"] = "MILP mejorado"

    rule_text = row_text(["regla", "multinivel"], default=params["multilevel_rule"])
    rule_key = normalize_text(rule_text)
    if "perimetro" in rule_key:
        params["multilevel_rule"] = "Según perímetro"
    elif "porcentaje" in rule_key or "estandar" in rule_key:
        params["multilevel_rule"] = "Porcentaje de puntos estándar"
    elif "fijo" in rule_key or "circulo" in rule_key:
        params["multilevel_rule"] = "Fijo por círculo"

    params["multilevel_pct"] = row_value(["porcentaje", "multinivel"], default=params["multilevel_pct"])
    params["ga_generations"] = int(row_value(["generaciones", "genetico"], default=params["ga_generations"]))
    params["ga_population"] = int(row_value(["poblacion", "genetico"], default=params["ga_population"]))

    sep_text = row_text(["separaciones"], contains_any=["permitidas", "puntos"], default=params["separations_text"])
    if sep_text:
        params["separations_text"] = sep_text

    expected = [
        "total_days", "hours_per_day", "t_standard", "t_multilevel", "t_24h", "t_extended",
        "budget", "daily_cost", "min_multilevel_per_circle", "min_24h", "min_extended",
        "max_teams", "w_standard", "w_multilevel", "w_24h", "w_extended"
    ]
    missing = [k for k in expected if params.get(k) is None]
    if missing:
        warnings.append("Algunos parámetros no fueron reconocidos y se usaron valores por defecto: " + ", ".join(missing))

    return params, params_table.drop(columns=["Clave normalizada"]), lab_df, warnings


# -----------------------------
# Modelo MILP
# -----------------------------

def points_from_perimeter(perimeter: float, separation: float) -> int:
    if separation <= 0:
        return 1
    return max(1, int(math.ceil(perimeter / separation)))



def min_multilevel_by_perimeter(perimeter: float) -> int:
    """Regla adaptativa para el MILP mejorado."""
    if perimeter < 200:
        return 1
    if perimeter < 500:
        return 2
    if perimeter < 1000:
        return 3
    return 4


def compute_plan_metrics(
    circles_df: pd.DataFrame,
    separations_selected: Dict[str, float],
    multilevel_selected: Dict[str, int],
    lab_samples: Dict[str, int],
    y_24h: int,
    y_ext: int,
    teams: int,
    hours_per_day: float,
    daily_cost: float,
    t_standard: float,
    t_multilevel: float,
    t_24h: float,
    t_extended: float,
    w_standard: float,
    w_multilevel: float,
    w_24h: float,
    w_extended: float,
    w_lab: Dict[str, float],
    lab_costs: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    rows = []
    standard_total = 0
    multilevel_total = 0
    for _, row in circles_df.iterrows():
        c = str(row["Círculo"])
        p = float(row["Perímetro (m)"])
        s = float(separations_selected[c])
        pts = points_from_perimeter(p, s)
        mpts = int(multilevel_selected[c])
        standard_total += pts
        multilevel_total += mpts
        rows.append({
            "Círculo": c,
            "Perímetro (m)": p,
            "Separación seleccionada (m)": s,
            "Puntos estándar": pts,
            "Puntos multinivel": mpts,
            "Total puntos": pts + mpts,
        })

    strategy_df = pd.DataFrame(rows)
    lab_df = pd.DataFrame({
        "Tipo de análisis": list(lab_samples.keys()),
        "Muestras": [int(lab_samples[k]) for k in lab_samples],
        "Costo unitario": [lab_costs[k] for k in lab_samples],
        "Costo total": [int(lab_samples[k]) * lab_costs[k] for k in lab_samples],
    })

    total_h = (
        t_standard * standard_total
        + t_multilevel * multilevel_total
        + t_24h * y_24h
        + t_extended * y_ext
    )
    operation_days = total_h / max(hours_per_day, 1e-6)
    calendar_days_est = total_h / max(hours_per_day * teams, 1e-6)
    lab_total = float(lab_df["Costo total"].sum())
    operation_total = operation_days * daily_cost
    quality = (
        w_standard * standard_total
        + w_multilevel * multilevel_total
        + w_24h * y_24h
        + w_extended * y_ext
        + sum(w_lab[k] * lab_samples[k] for k in lab_samples)
    )
    metrics = {
        "Calidad científica": float(quality),
        "Horas de trabajo": float(total_h),
        "Días-equipo operativos": float(operation_days),
        "Días calendario estimados": float(calendar_days_est),
        "Equipos": int(teams),
        "Mediciones 24 h": int(y_24h),
        "Mediciones extendidas": int(y_ext),
        "Costo operativo": float(operation_total),
        "Costo laboratorio": float(lab_total),
        "Costo total": float(operation_total + lab_total),
    }
    return strategy_df, lab_df, metrics

def solve_model(
    circles_df: pd.DataFrame,
    separations: List[int],
    lab_types: List[str],
    total_days: float,
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
    model_variant: str = "MILP actual",
    multilevel_rule: str = "Fijo por círculo",
    multilevel_pct: float = 0.10,
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
    h_total = pulp.LpVariable("horas_trabajo", lowBound=0, cat="Continuous")
    y_24h = pulp.LpVariable("mediciones_24h", lowBound=0, cat="Integer")
    y_ext = pulp.LpVariable("mediciones_extendidas", lowBound=0, cat="Integer")
    lab = pulp.LpVariable.dicts("muestras_lab", lab_types, lowBound=0, cat="Integer")

    standard_points_expr = pulp.lpSum(n_points[(i, s)] * x[i][s] for i in circles for s in separations)
    multilevel_expr = pulp.lpSum(m[i] for i in circles)

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
    model += h_total <= total_days * hours_per_day * teams, "R2_capacidad_operativa"

    # Aproximación lineal: costo operativo por día de trabajo efectivo de un equipo.
    operation_days_expr = h_total / max(hours_per_day, 1e-6)
    operation_cost_expr = operation_days_expr * daily_cost
    lab_cost_expr = pulp.lpSum(lab_costs[k] * lab[k] for k in lab_types)
    model += operation_cost_expr + lab_cost_expr <= budget, "R3_presupuesto"

    for i in circles:
        model += pulp.lpSum(x[i][s] for s in separations) == 1, f"R4_separacion_unica_{i}"

        selected_standard_points_i = pulp.lpSum(n_points[(i, s)] * x[i][s] for s in separations)

        if model_variant == "MILP mejorado" and multilevel_rule == "Según perímetro":
            model += m[i] >= min_multilevel_by_perimeter(perimeter[i]), f"R7_min_multinivel_perimetro_{i}"
        elif model_variant == "MILP mejorado" and multilevel_rule == "Porcentaje de puntos estándar":
            model += m[i] >= multilevel_pct * selected_standard_points_i, f"R7_min_multinivel_porcentaje_{i}"
        else:
            model += m[i] >= min_multilevel_per_circle, f"R7_min_multinivel_fijo_{i}"

        # Límite superior para evitar soluciones artificialmente intensivas.
        if model_variant == "MILP mejorado":
            model += m[i] <= selected_standard_points_i, f"R7_max_multinivel_{i}"

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
        selected_teams = int(round(pulp.value(teams)))
        operation_days = total_h / max(hours_per_day, 1e-6)
        calendar_days_est = total_h / max(hours_per_day * selected_teams, 1e-6)
        lab_total = float(sum(lab_df["Costo total"]))
        operation_total = operation_days * daily_cost
        metrics = {
            "Calidad científica": float(pulp.value(model.objective)),
            "Horas de trabajo": total_h,
            "Días-equipo operativos": operation_days,
            "Días calendario estimados": calendar_days_est,
            "Equipos": selected_teams,
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


def solve_genetic(
    circles_df: pd.DataFrame,
    separations: List[int],
    lab_types: List[str],
    total_days: float,
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
    generations: int = 120,
    population_size: int = 80,
    mutation_rate: float = 0.12,
    multilevel_rule: str = "Según perímetro",
    multilevel_pct: float = 0.10,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Algoritmo genético simple sin librerías externas.

    Nota: es una heurística. Busca una solución buena, pero no garantiza el óptimo global.
    """
    rng = np.random.default_rng(42)
    circles = circles_df["Círculo"].astype(str).tolist()
    perimeter = dict(zip(circles, circles_df["Perímetro (m)"].astype(float)))
    max_hours = total_days * hours_per_day * max_teams

    def min_m_for(c: str, sep: float) -> int:
        std = points_from_perimeter(perimeter[c], sep)
        if multilevel_rule == "Según perímetro":
            return min_multilevel_by_perimeter(perimeter[c])
        if multilevel_rule == "Porcentaje de puntos estándar":
            return max(1, int(math.ceil(multilevel_pct * std)))
        return int(min_multilevel_per_circle)

    def random_individual():
        sep_idx = rng.integers(0, len(separations), size=len(circles)).tolist()
        mvals = []
        for c, idx in zip(circles, sep_idx):
            sep = separations[idx]
            std = points_from_perimeter(perimeter[c], sep)
            mn = min_m_for(c, sep)
            mx = max(mn, std)
            mvals.append(int(rng.integers(mn, mx + 1)))
        y24 = int(rng.integers(min_24h, max(min_24h + 1, min_24h + 6)))
        yext = int(rng.integers(min_extended, max(min_extended + 1, min_extended + 4)))
        teams = int(rng.integers(1, max_teams + 1))
        return {"sep_idx": sep_idx, "m": mvals, "y24": y24, "yext": yext, "teams": teams}

    def repair(ind):
        ind["sep_idx"] = [int(np.clip(v, 0, len(separations) - 1)) for v in ind["sep_idx"]]
        ind["teams"] = int(np.clip(ind["teams"], 1, max_teams))
        ind["y24"] = max(int(ind["y24"]), int(min_24h))
        ind["yext"] = max(int(ind["yext"]), int(min_extended))
        fixed_m = []
        for c, idx, mv in zip(circles, ind["sep_idx"], ind["m"]):
            sep = separations[idx]
            std = points_from_perimeter(perimeter[c], sep)
            mn = min_m_for(c, sep)
            fixed_m.append(int(np.clip(mv, mn, max(mn, std))))
        ind["m"] = fixed_m
        return ind

    def evaluate(ind):
        ind = repair(ind.copy())
        sep_sel = {c: separations[idx] for c, idx in zip(circles, ind["sep_idx"])}
        m_sel = {c: mv for c, mv in zip(circles, ind["m"])}
        lab_samples = {k: int(min_lab_samples[k]) for k in lab_types}
        strategy_df, lab_df, metrics = compute_plan_metrics(
            circles_df, sep_sel, m_sel, lab_samples, ind["y24"], ind["yext"], ind["teams"],
            hours_per_day, daily_cost, t_standard, t_multilevel, t_24h, t_extended,
            w_standard, w_multilevel, w_24h, w_extended, w_lab, lab_costs,
        )
        penalty = 0.0
        if metrics["Horas de trabajo"] > max_hours:
            penalty += 1000.0 * (metrics["Horas de trabajo"] - max_hours)
        if metrics["Costo total"] > budget:
            penalty += 0.00001 * (metrics["Costo total"] - budget)
        # Penalización leve por días calendario excesivos respecto al horizonte disponible.
        if metrics["Días calendario estimados"] > total_days:
            penalty += 100.0 * (metrics["Días calendario estimados"] - total_days)
        return metrics["Calidad científica"] - penalty, strategy_df, lab_df, metrics

    def crossover(a, b):
        cut = int(rng.integers(1, max(2, len(circles))))
        child = {
            "sep_idx": a["sep_idx"][:cut] + b["sep_idx"][cut:],
            "m": a["m"][:cut] + b["m"][cut:],
            "y24": a["y24"] if rng.random() < 0.5 else b["y24"],
            "yext": a["yext"] if rng.random() < 0.5 else b["yext"],
            "teams": a["teams"] if rng.random() < 0.5 else b["teams"],
        }
        return repair(child)

    def mutate(ind):
        ind = ind.copy()
        ind["sep_idx"] = list(ind["sep_idx"])
        ind["m"] = list(ind["m"])
        for j in range(len(circles)):
            if rng.random() < mutation_rate:
                ind["sep_idx"][j] = int(rng.integers(0, len(separations)))
            if rng.random() < mutation_rate:
                ind["m"][j] += int(rng.integers(-2, 3))
        if rng.random() < mutation_rate:
            ind["y24"] += int(rng.integers(-2, 3))
        if rng.random() < mutation_rate:
            ind["yext"] += int(rng.integers(-1, 2))
        if rng.random() < mutation_rate:
            ind["teams"] += int(rng.integers(-1, 2))
        return repair(ind)

    population = [random_individual() for _ in range(population_size)]
    best = None
    best_score = -1e30

    for _ in range(generations):
        scored = []
        for ind in population:
            score, _, _, _ = evaluate(ind)
            scored.append((score, ind))
            if score > best_score:
                best_score = score
                best = ind
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [ind for _, ind in scored[: max(2, population_size // 5)]]
        new_pop = elites.copy()
        while len(new_pop) < population_size:
            parents_idx = rng.choice(len(elites), size=2, replace=True)
            child = crossover(elites[int(parents_idx[0])], elites[int(parents_idx[1])])
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    score, strategy_df, lab_df, metrics = evaluate(best)
    feasible = metrics["Horas de trabajo"] <= max_hours and metrics["Costo total"] <= budget and metrics["Días calendario estimados"] <= total_days
    metrics["Fitness genético"] = float(score)
    metrics["Margen presupuesto"] = budget - metrics["Costo total"]
    status = "Heurística factible" if feasible else "Heurística no factible"
    return status, strategy_df, lab_df, metrics


def to_excel(strategy_df: pd.DataFrame, lab_df: pd.DataFrame, metrics: Dict[str, float], params_table: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        strategy_df.to_excel(writer, index=False, sheet_name="Estrategia")
        lab_df.to_excel(writer, index=False, sheet_name="Laboratorio")
        pd.DataFrame(list(metrics.items()), columns=["Concepto", "Valor"]).to_excel(
            writer, index=False, sheet_name="Resumen"
        )
        if not params_table.empty:
            params_table.to_excel(writer, index=False, sheet_name="Parametros cargados")

        workbook = writer.book
        money_fmt = workbook.add_format({"num_format": "$#,##0"})
        num_fmt = workbook.add_format({"num_format": "0.00"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.set_row(0, None, header_fmt)
            ws.set_column(0, 0, 24)
            ws.set_column(1, 10, 20)

        writer.sheets["Laboratorio"].set_column(2, 3, 18, money_fmt)
        writer.sheets["Resumen"].set_column(1, 1, 20, num_fmt)

    return output.getvalue()



def render_solution(title: str, status: str, strategy_df: pd.DataFrame, lab_df: pd.DataFrame, metrics: Dict[str, float], params_table: pd.DataFrame):
    st.markdown(f"## {title}")
    if status not in ["Optimal", "Feasible", "Heurística factible"]:
        st.error(f"No se encontró solución factible. Estado: {status}")
        st.info("Prueba aumentar presupuesto/tiempo, reducir mínimos, usar separaciones mayores o permitir más equipos.")
        return

    st.success(f"Solución encontrada: {status}")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Calidad científica", f"{metrics['Calidad científica']:.4f}")
    kpi2.metric("Costo total", f"${metrics['Costo total']:,.0f}")
    kpi3.metric("Margen presupuesto", f"${metrics.get('Margen presupuesto', 0):,.0f}")
    kpi4.metric("Días calendario estimados", f"{metrics['Días calendario estimados']:.1f}")

    st.markdown("### Estrategia")
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
    st.markdown("### Laboratorio")
    st.dataframe(lab_df, use_container_width=True, hide_index=True)
    st.markdown("### Resumen")
    st.json(metrics)

    excel_bytes = to_excel(strategy_df, lab_df, metrics, params_table)
    st.download_button(
        f"Descargar resultados en Excel — {title}",
        data=excel_bytes,
        file_name=f"resultados_{title.lower().replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# -----------------------------
# Interfaz Streamlit
# -----------------------------
st.title("🧪 Optimizador MILP de estrategia de muestreo")
st.caption(
    "Modelo de programación lineal entera mixta para maximizar calidad científica bajo restricciones de tiempo, presupuesto y mínimos técnicos."
)

# Valores base. Se actualizan inmediatamente si el usuario sube el Excel de parámetros,
# para que también puedan controlar el método, la regla multinivel, las separaciones y el algoritmo genético.
params = DEFAULT_PARAMS.copy()
lab_config_base = DEFAULT_LAB_CONFIG.copy()
params_table = pd.DataFrame()
param_warnings = []
param_source = "Valores por defecto"

with st.sidebar:
    st.header("1. Archivos de entrada")

    uploaded_params_file = st.file_uploader(
        "Subir Excel de parámetros de estrategia",
        type=["xlsx", "xls"],
        help="Formato esperado: columna A = parámetro, columna B = unidad, columna C = valor.",
    )

    if uploaded_params_file is not None:
        try:
            params, params_table, lab_config_base, param_warnings = read_strategy_parameters_excel(uploaded_params_file)
            param_source = "Excel de parámetros subido"
        except Exception as exc:
            st.error(f"No pude leer el Excel de parámetros: {exc}")
            st.info("Usa el formato: columna A = parámetro, columna B = unidad, columna C = valor.")

    uploaded_circles_file = st.file_uploader(
        "Subir Excel con círculos de hadas",
        type=["xlsx", "xls"],
        help="Formato esperado: columna ID y columna Perimetro (m). También acepta Perímetro (m).",
    )

    use_example = st.checkbox("Usar Excel de ejemplo si no subo círculos", value=True)
    n_circles = st.number_input(
        "Número de círculos para modo manual/ejemplo",
        min_value=1,
        max_value=500,
        value=30,
        step=1,
    )
    allow_edit_circles = st.checkbox("Permitir editar la tabla de círculos", value=False)

    st.header("2. Separaciones permitidas")
    sep_text = st.text_input("Separaciones en metros", value=str(params.get("separations_text", ", ".join(map(str, DEFAULT_SEPARATIONS)))))

    st.header("3. Modelo de optimización")
    model_options = [
        "MILP actual",
        "MILP mejorado",
        "Algoritmo genético",
        "Comparar MILP mejorado vs genético",
    ]
    default_model = params.get("model_choice", "Comparar MILP mejorado vs genético")
    model_choice = st.selectbox(
        "Selecciona el modelo",
        model_options,
        index=model_options.index(default_model) if default_model in model_options else 3,
    )

    rule_options = ["Fijo por círculo", "Según perímetro", "Porcentaje de puntos estándar"]
    default_rule = params.get("multilevel_rule", "Según perímetro")
    multilevel_rule = st.selectbox(
        "Regla para puntos multinivel",
        rule_options,
        index=rule_options.index(default_rule) if default_rule in rule_options else 1,
    )

    multilevel_pct = st.number_input(
        "Porcentaje multinivel sobre puntos estándar",
        min_value=0.0,
        max_value=1.0,
        value=float(params.get("multilevel_pct", 0.10)),
        step=0.01,
        help="Solo se usa si eliges la regla porcentual. También puede venir desde el Excel de parámetros.",
    )
    ga_generations = st.number_input(
        "Generaciones algoritmo genético",
        min_value=20,
        max_value=500,
        value=int(params.get("ga_generations", 120)),
        step=20,
    )
    ga_population = st.number_input(
        "Población algoritmo genético",
        min_value=20,
        max_value=300,
        value=int(params.get("ga_population", 80)),
        step=20,
    )

st.subheader("A. Parámetros de estrategia")

pcol1, pcol2 = st.columns([1, 3])
with pcol1:
    st.metric("Fuente de parámetros", param_source)
    if not params_table.empty:
        st.metric("Parámetros leídos", len(params_table))
with pcol2:
    if not params_table.empty:
        st.dataframe(
            params_table[["Sección", "Parámetro", "Unidad", "Valor"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Puedes subir el Excel de parámetros o usar los valores por defecto.")

for msg in param_warnings:
    st.warning(msg)

st.subheader("B. Geometría de los círculos")

uploaded_error = None
if uploaded_circles_file is not None:
    try:
        base_df = normalize_circles_excel(uploaded_circles_file)
        data_source = "Archivo de círculos subido"
    except Exception as exc:
        uploaded_error = str(exc)
        base_df = pd.DataFrame({"Círculo": [], "Perímetro (m)": []})
        data_source = "Archivo no válido"
elif use_example:
    base_df = load_example_data().head(int(n_circles)).copy()
    if len(base_df) < n_circles:
        mean_perimeter = float(base_df["Perímetro (m)"].mean()) if not base_df.empty else 100.0
        extra = pd.DataFrame(
            {
                "Círculo": [f"C{i}" for i in range(len(base_df) + 1, int(n_circles) + 1)],
                "Perímetro (m)": np.repeat(mean_perimeter, int(n_circles) - len(base_df)),
            }
        )
        base_df = pd.concat([base_df, extra], ignore_index=True)
    data_source = "Excel de ejemplo"
else:
    base_df = pd.DataFrame(
        {"Círculo": [f"C{i}" for i in range(1, int(n_circles) + 1)], "Perímetro (m)": [100.0] * int(n_circles)}
    )
    data_source = "Tabla manual"

if uploaded_error:
    st.error(f"No pude leer el Excel de círculos: {uploaded_error}")
    st.info("Usa un archivo con dos columnas: ID y Perimetro (m). Ejemplo: C1 | 113")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Número de círculos de hadas", len(base_df))
metric_col2.metric("Perímetro total (m)", f"{base_df['Perímetro (m)'].sum():,.1f}" if not base_df.empty else "0")
metric_col3.metric("Fuente de círculos", data_source)

if allow_edit_circles:
    circles_df = st.data_editor(
        base_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"Perímetro (m)": st.column_config.NumberColumn(min_value=1.0, step=1.0)},
    )
else:
    circles_df = base_df.copy()
    st.dataframe(circles_df, use_container_width=True, hide_index=True)

st.subheader("C. Tiempo, presupuesto y equipos")

c1, c2, c3, c4 = st.columns(4)
with c1:
    total_days = st.number_input("Tiempo total disponible (días)", min_value=1.0, value=float(params["total_days"]), step=1.0)
with c2:
    hours_per_day = st.number_input("Horas efectivas por día", min_value=1.0, value=float(params["hours_per_day"]), step=1.0)
with c3:
    max_teams = st.number_input("Equipos/sensores disponibles", min_value=1, max_value=100, value=int(params["max_teams"]), step=1)
with c4:
    budget = st.number_input("Presupuesto total", min_value=0.0, value=float(params["budget"]), step=1_000_000.0)

c5, c6, c7, c8, c9 = st.columns(5)
with c5:
    daily_cost = st.number_input("Costo diario de muestreo", min_value=0.0, value=float(params["daily_cost"]), step=50_000.0)
with c6:
    t_standard = st.number_input("Tiempo punto estándar (h)", min_value=0.0, value=float(params["t_standard"]), step=0.25)
with c7:
    t_multilevel = st.number_input("Tiempo punto multinivel (h)", min_value=0.0, value=float(params["t_multilevel"]), step=0.25)
with c8:
    t_24h = st.number_input("Tiempo medición 24 h (h)", min_value=0.0, value=float(params["t_24h"]), step=1.0)
with c9:
    t_extended = st.number_input("Tiempo medición extendida (h)", min_value=0.0, value=float(params["t_extended"]), step=1.0)

st.subheader("D. Requisitos mínimos")

r1, r2, r3 = st.columns(3)
with r1:
    min_multilevel_per_circle = st.number_input(
        "Puntos multinivel mínimos por círculo", min_value=0, value=int(params["min_multilevel_per_circle"]), step=1
    )
with r2:
    min_24h = st.number_input("Mediciones 24 h mínimas", min_value=0, value=int(params["min_24h"]), step=1)
with r3:
    min_extended = st.number_input("Mediciones extendidas mínimas", min_value=0, value=int(params["min_extended"]), step=1)

st.subheader("E. Análisis de laboratorio")
lab_config = st.data_editor(
    lab_config_base,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Mínimo de muestras": st.column_config.NumberColumn(min_value=0, step=1),
        "Costo unitario": st.column_config.NumberColumn(min_value=0.0, step=10_000.0),
        "Peso calidad": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"),
    },
)

st.subheader("F. Pesos de calidad científica")
col1, col2, col3, col4 = st.columns(4)
with col1:
    w_standard = st.number_input("Peso punto estándar", min_value=0.0, value=float(params["w_standard"]), step=0.0001, format="%.4f")
with col2:
    w_multilevel = st.number_input("Peso punto multinivel", min_value=0.0, value=float(params["w_multilevel"]), step=0.0001, format="%.4f")
with col3:
    w_24h = st.number_input("Peso medición 24 h", min_value=0.0, value=float(params["w_24h"]), step=0.0001, format="%.4f")
with col4:
    w_extended = st.number_input("Peso medición extendida", min_value=0.0, value=float(params["w_extended"]), step=0.0001, format="%.4f")

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
        common_kwargs = dict(
            circles_df=circles_df,
            separations=separations,
            lab_types=lab_types,
            total_days=float(total_days),
            hours_per_day=float(hours_per_day),
            budget=float(budget),
            daily_cost=float(daily_cost),
            t_standard=float(t_standard),
            t_multilevel=float(t_multilevel),
            t_24h=float(t_24h),
            t_extended=float(t_extended),
            min_multilevel_per_circle=int(min_multilevel_per_circle),
            min_24h=int(min_24h),
            min_extended=int(min_extended),
            min_lab_samples=min_lab_samples,
            max_teams=int(max_teams),
            w_standard=float(w_standard),
            w_multilevel=float(w_multilevel),
            w_24h=float(w_24h),
            w_extended=float(w_extended),
            w_lab=w_lab,
            lab_costs=lab_costs,
        )

        if model_choice == "MILP actual":
            status, strategy_df, lab_df, metrics = solve_model(
                **common_kwargs,
                model_variant="MILP actual",
                multilevel_rule="Fijo por círculo",
                multilevel_pct=float(multilevel_pct),
            )
            render_solution("MILP actual", status, strategy_df, lab_df, metrics, params_table)

        elif model_choice == "MILP mejorado":
            status, strategy_df, lab_df, metrics = solve_model(
                **common_kwargs,
                model_variant="MILP mejorado",
                multilevel_rule=multilevel_rule,
                multilevel_pct=float(multilevel_pct),
            )
            render_solution("MILP mejorado", status, strategy_df, lab_df, metrics, params_table)

        elif model_choice == "Algoritmo genético":
            status, strategy_df, lab_df, metrics = solve_genetic(
                **common_kwargs,
                generations=int(ga_generations),
                population_size=int(ga_population),
                multilevel_rule=multilevel_rule,
                multilevel_pct=float(multilevel_pct),
            )
            render_solution("Algoritmo genético", status, strategy_df, lab_df, metrics, params_table)

        else:
            status_m, strategy_m, lab_m, metrics_m = solve_model(
                **common_kwargs,
                model_variant="MILP mejorado",
                multilevel_rule=multilevel_rule,
                multilevel_pct=float(multilevel_pct),
            )
            status_g, strategy_g, lab_g, metrics_g = solve_genetic(
                **common_kwargs,
                generations=int(ga_generations),
                population_size=int(ga_population),
                multilevel_rule=multilevel_rule,
                multilevel_pct=float(multilevel_pct),
            )

            comp = pd.DataFrame([
                {
                    "Modelo": "MILP mejorado",
                    "Estado": status_m,
                    "Calidad científica": metrics_m.get("Calidad científica"),
                    "Costo total": metrics_m.get("Costo total"),
                    "Horas de trabajo": metrics_m.get("Horas de trabajo"),
                    "Días calendario": metrics_m.get("Días calendario estimados"),
                    "Equipos": metrics_m.get("Equipos"),
                },
                {
                    "Modelo": "Algoritmo genético",
                    "Estado": status_g,
                    "Calidad científica": metrics_g.get("Calidad científica"),
                    "Costo total": metrics_g.get("Costo total"),
                    "Horas de trabajo": metrics_g.get("Horas de trabajo"),
                    "Días calendario": metrics_g.get("Días calendario estimados"),
                    "Equipos": metrics_g.get("Equipos"),
                },
            ])
            st.markdown("## Comparación de modelos")
            st.dataframe(comp, use_container_width=True, hide_index=True)

            col_a, col_b = st.columns(2)
            with col_a:
                render_solution("MILP mejorado", status_m, strategy_m, lab_m, metrics_m, params_table)
            with col_b:
                render_solution("Algoritmo genético", status_g, strategy_g, lab_g, metrics_g, params_table)
else:
    st.info("Carga los archivos, ajusta los parámetros si lo necesitas y presiona **Resolver modelo**.")
