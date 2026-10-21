# 🚢 EDA Titanic

App interactiva en [Streamlit](https://streamlit.io) para hacer un análisis exploratorio (EDA) del dataset del Titanic publicado en [`seaborn-data`](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv).

## Qué incluye

- **Filtros** (barra lateral): clase, sexo, puerto de embarque, rango de edad y de tarifa, con opción de incluir o excluir valores desconocidos. Todos los análisis se recalculan con la selección.
- **Datos**: vista previa, estadísticas descriptivas y descarga del CSV filtrado.
- **Calidad de datos**: nulos por columna (tabla, gráfico y mapa de nulos), filas duplicadas, tarifas en 0 y columnas redundantes (`survived`/`alive`, `pclass`/`class`, `embarked`/`embark_town`).
- **Distribuciones**: histogramas con boxplot marginal, boxplots por clase, estadísticos (media, mediana, asimetría, outliers por IQR) y conteos de variables categóricas.
- **Supervivencia**: tasa por grupo, mapa de calor clase × sexo y supervivencia por edad.
- **Correlaciones**: matriz de Pearson, Spearman o Kendall, ranking de correlación con `survived` y diagrama de dispersión.

## Cómo correrla

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

La app se abre en http://localhost:8501. Necesita conexión a internet para descargar el CSV (se cachea tras la primera carga).
