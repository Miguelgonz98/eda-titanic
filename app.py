"""EDA interactivo del dataset del Titanic con Streamlit."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

st.set_page_config(page_title="EDA Titanic", page_icon="🚢", layout="wide")


@st.cache_data(show_spinner="Descargando datos...")
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def survival_rate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    out = (
        df.groupby(by, observed=True, dropna=False)["survived"]
        .agg(pasajeros="count", sobrevivientes="sum", tasa="mean")
        .reset_index()
    )
    out[by] = out[by].astype(str)
    return out


try:
    raw = load_data(DATA_URL)
except Exception as exc:  # red caída, URL inválida, etc.
    st.error(f"No se pudo descargar el dataset: {exc}")
    st.stop()

# ---------------------------------------------------------------- Filtros
st.sidebar.header("Filtros")

classes = sorted(raw["pclass"].dropna().unique())
sel_class = st.sidebar.multiselect("Clase", classes, default=classes)

sexes = sorted(raw["sex"].dropna().unique())
sel_sex = st.sidebar.multiselect("Sexo", sexes, default=sexes)

ports = sorted(raw["embark_town"].dropna().unique())
sel_port = st.sidebar.multiselect("Puerto de embarque", ports, default=ports)
include_na_port = st.sidebar.checkbox("Incluir puerto desconocido", value=True)

age_min, age_max = float(raw["age"].min()), float(raw["age"].max())
sel_age = st.sidebar.slider("Edad", age_min, age_max, (age_min, age_max), step=1.0)
include_na_age = st.sidebar.checkbox("Incluir edad desconocida", value=True)

fare_max = float(raw["fare"].max())
sel_fare = st.sidebar.slider("Tarifa", 0.0, fare_max, (0.0, fare_max), step=1.0)

mask = (
    raw["pclass"].isin(sel_class)
    & raw["sex"].isin(sel_sex)
    & (raw["embark_town"].isin(sel_port) | (include_na_port & raw["embark_town"].isna()))
    & (raw["age"].between(*sel_age) | (include_na_age & raw["age"].isna()))
    & raw["fare"].between(*sel_fare)
)
df = raw[mask]

# ---------------------------------------------------------------- Encabezado
st.title("🚢 EDA del Titanic")
st.caption(f"Fuente: [seaborn-data/titanic.csv]({DATA_URL})")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pasajeros", f"{len(df):,}", f"{len(df) - len(raw):,} vs total" if len(df) != len(raw) else None)
c2.metric("Tasa de supervivencia", f"{df['survived'].mean():.1%}" if len(df) else "—")
c3.metric("Edad media", f"{df['age'].mean():.1f}" if df["age"].notna().any() else "—")
c4.metric("Tarifa media", f"{df['fare'].mean():.2f}" if len(df) else "—")

if df.empty:
    st.warning("Ningún pasajero cumple los filtros seleccionados.")
    st.stop()

tab_data, tab_quality, tab_dist, tab_surv, tab_corr = st.tabs(
    ["Datos", "Calidad de datos", "Distribuciones", "Supervivencia", "Correlaciones"]
)

# ---------------------------------------------------------------- Datos
with tab_data:
    st.subheader("Vista previa")
    st.dataframe(df, width="stretch", height=400)
    st.download_button(
        "Descargar CSV filtrado",
        df.to_csv(index=False).encode("utf-8"),
        file_name="titanic_filtrado.csv",
        mime="text/csv",
    )
    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe(include="all").T.astype(str).replace("nan", ""), width="stretch")

# ---------------------------------------------------------------- Calidad
with tab_quality:
    st.subheader("Valores faltantes")
    nulls = pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "% nulos": (df.isna().mean() * 100).round(2),
            "valores únicos": df.nunique(),
        }
    ).sort_values("% nulos", ascending=False)
    col_a, col_b = st.columns([1, 1])
    col_a.dataframe(nulls, width="stretch")
    missing = nulls[nulls["nulos"] > 0].reset_index(names="columna")
    if missing.empty:
        col_b.success("No hay valores faltantes en la selección.")
    else:
        col_b.plotly_chart(
            px.bar(missing, x="columna", y="% nulos", title="% de valores faltantes por columna"),
            width="stretch",
        )

    st.subheader("Mapa de nulos")
    st.plotly_chart(
        px.imshow(
            df.isna().T.astype(int),
            aspect="auto",
            color_continuous_scale=["#e8eef7", "#d6604d"],
            labels={"x": "fila", "y": "columna", "color": "nulo"},
        ),
        width="stretch",
    )

    st.subheader("Duplicados y consistencia")
    dups = int(df.duplicated().sum())
    q1, q2, q3 = st.columns(3)
    q1.metric("Filas duplicadas", dups, help="Filas idénticas en todas las columnas")
    q2.metric("Tarifas en 0", int((df["fare"] == 0).sum()))
    redundant = {
        "survived ↔ alive": (df["survived"] == (df["alive"] == "yes").astype(int)).all(),
        "pclass ↔ class": (df["pclass"].map({1: "First", 2: "Second", 3: "Third"}) == df["class"]).all(),
        "embarked ↔ embark_town": (
            df["embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
            .fillna("NA")
            .eq(df["embark_town"].fillna("NA"))
            .all()
        ),
    }
    q3.metric("Pares de columnas redundantes", sum(redundant.values()), help="Columnas que codifican la misma información")
    st.dataframe(
        pd.DataFrame({"par": list(redundant), "coinciden 100%": list(redundant.values())}),
        hide_index=True,
    )
    if dups:
        st.info(
            "Los duplicados probablemente son pasajeros distintos con el mismo perfil: "
            "el dataset no incluye nombre ni ID de pasajero."
        )

# ---------------------------------------------------------------- Distribuciones
with tab_dist:
    numeric_cols = ["age", "fare", "sibsp", "parch"]
    cat_cols = ["pclass", "sex", "embark_town", "who", "deck", "alone", "adult_male"]

    st.subheader("Variables numéricas")
    num_col = st.selectbox("Variable", numeric_cols, key="num")
    bins = st.slider("Bins", 5, 80, 30)
    d1, d2 = st.columns(2)
    d1.plotly_chart(
        px.histogram(df, x=num_col, nbins=bins, marginal="box", title=f"Histograma de {num_col}"),
        width="stretch",
    )
    d2.plotly_chart(
        px.box(df, x="pclass", y=num_col, points="outliers", title=f"{num_col} por clase"),
        width="stretch",
    )
    s = df[num_col].dropna()
    st.caption(
        f"Media {s.mean():.2f} · Mediana {s.median():.2f} · Desv. est. {s.std():.2f} · "
        f"Asimetría {s.skew():.2f} · Outliers (IQR) {int(((s < s.quantile(.25) - 1.5 * (s.quantile(.75) - s.quantile(.25))) | (s > s.quantile(.75) + 1.5 * (s.quantile(.75) - s.quantile(.25)))).sum())}"
    )

    st.subheader("Variables categóricas")
    cat_col = st.selectbox("Variable", cat_cols, key="cat")
    counts = df[cat_col].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [cat_col, "pasajeros"]
    st.plotly_chart(px.bar(counts, x=cat_col, y="pasajeros", title=f"Conteo por {cat_col}"), width="stretch")

# ---------------------------------------------------------------- Supervivencia
with tab_surv:
    st.subheader("Tasa de supervivencia por grupo")
    group = st.selectbox("Agrupar por", ["sex", "pclass", "who", "embark_town", "deck", "alone", "sibsp", "parch"])
    rates = survival_rate(df, group)
    s1, s2 = st.columns([2, 1])
    s1.plotly_chart(
        px.bar(rates, x=group, y="tasa", text_auto=".0%", hover_data=["pasajeros", "sobrevivientes"],
               title=f"Supervivencia por {group}").update_yaxes(tickformat=".0%", range=[0, 1]),
        width="stretch",
    )
    s2.dataframe(rates.style.format({"tasa": "{:.1%}"}), hide_index=True, width="stretch")

    st.subheader("Clase × sexo")
    pivot = df.pivot_table(index="sex", columns="pclass", values="survived", aggfunc="mean")
    st.plotly_chart(
        px.imshow(pivot, text_auto=".0%", color_continuous_scale="RdBu", zmin=0, zmax=1,
                  labels={"color": "supervivencia"}),
        width="stretch",
    )

    st.subheader("Supervivencia por edad")
    age_df = df.dropna(subset=["age"]).copy()
    age_df["grupo_edad"] = pd.cut(age_df["age"], bins=[0, 12, 18, 30, 45, 60, 100],
                                  labels=["0-12", "13-18", "19-30", "31-45", "46-60", "60+"])
    a1, a2 = st.columns(2)
    a1.plotly_chart(
        px.histogram(age_df, x="age", color=age_df["survived"].map({0: "No", 1: "Sí"}), nbins=40,
                     barmode="overlay", opacity=0.6, labels={"color": "Sobrevivió"},
                     title="Distribución de edad por supervivencia"),
        width="stretch",
    )
    age_rates = survival_rate(age_df, "grupo_edad")
    a2.plotly_chart(
        px.bar(age_rates, x="grupo_edad", y="tasa", text_auto=".0%", title="Supervivencia por grupo de edad")
        .update_yaxes(tickformat=".0%", range=[0, 1]),
        width="stretch",
    )

# ---------------------------------------------------------------- Correlaciones
with tab_corr:
    st.subheader("Matriz de correlación")
    method = st.radio("Método", ["pearson", "spearman", "kendall"], horizontal=True)
    corr_df = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].copy()
    corr_df["sex_male"] = (df["sex"] == "male").astype(int)
    corr_df["alone"] = df["alone"].astype(int)
    corr_df["adult_male"] = df["adult_male"].astype(int)
    corr = corr_df.corr(method=method)
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"),
        width="stretch",
    )
    st.caption("Las variables categóricas binarias se codificaron como 0/1 (sex_male, alone, adult_male).")

    st.subheader("Correlación con `survived`")
    target = corr["survived"].drop("survived").sort_values(key=np.abs, ascending=False).reset_index()
    target.columns = ["variable", "correlación"]
    st.plotly_chart(
        px.bar(target, x="correlación", y="variable", orientation="h", color="correlación",
               color_continuous_scale="RdBu_r", range_color=[-1, 1]).update_yaxes(autorange="reversed"),
        width="stretch",
    )

    st.subheader("Dispersión")
    x_col = st.selectbox("Eje X", ["age", "fare", "sibsp", "parch"], index=0)
    y_col = st.selectbox("Eje Y", ["fare", "age", "sibsp", "parch"], index=0)
    st.plotly_chart(
        px.scatter(df, x=x_col, y=y_col, color=df["survived"].map({0: "No", 1: "Sí"}),
                   labels={"color": "Sobrevivió"}, opacity=0.7),
        width="stretch",
    )
