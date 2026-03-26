import streamlit as st
from catboost import CatBoostClassifier
from utils import predecir_caso_personalizado

# =========================
# CARGAR MODELOS
# =========================
@st.cache_resource
def cargar_modelos():
    modelo_accion = CatBoostClassifier()
    modelo_accion.load_model("models/modelo_accion.cbm")
    
    modelo_estrategia = CatBoostClassifier()
    modelo_estrategia.load_model("models/modelo_estrategia.cbm")
    
    modelo_recurso = CatBoostClassifier()
    modelo_recurso.load_model("models/modelo_recurso.cbm")
    
    return modelo_accion, modelo_estrategia, modelo_recurso

modelo_accion, modelo_estrategia, modelo_recurso = cargar_modelos()

# =========================
# OPCIONES (AJUSTA SEGÚN TU DATASET)
# =========================
# =========================
# OPCIONES (DESDE COLAB)
# =========================

CURSOS = ["8vo", "9no", "10mo"]

BLOQUES = ["Algebra", "Geometria", "Estadistica"]

DESTREZAS = [
    "Resolver ecuaciones lineales",
    "Operar con fracciones",
    "Calcular areas",
    "Interpretar graficos",
    "Aplicar proporcionalidad"
]

TIPOS = [
    "Conceptual",
    "Procedimental",
    "Resolucion_problemas"
]

NIVELES = [1, 2, 3, 4, 5]

# =========================
# INTERFAZ
# =========================
st.set_page_config(page_title="Sistema de Recomendación", page_icon="📘")

st.title("📘 Sistema de Recomendación Docente")
st.markdown("Completa el formulario para generar una recomendación pedagógica.")

with st.form("formulario"):
    
    nombre = st.text_input("👤 Nombre del estudiante")

    curso = st.selectbox("Curso", CURSOS)
    bloque = st.selectbox("Bloque", BLOQUES)
    destreza = st.selectbox("Destreza evaluada", DESTREZAS)

    nivel = st.selectbox("Nivel de logro docente", NIVELES)
    complejidad = st.selectbox("Complejidad de la destreza", NIVELES)
    carga = st.selectbox("Carga cognitiva", NIVELES)

    tipo = st.selectbox("Tipo de competencia", TIPOS)

    submit = st.form_submit_button("🔍 Generar recomendación")

# =========================
# PREDICCIÓN
# =========================
if submit:
    
    caso = {
        "curso": curso,
        "bloque": bloque,
        "destreza": destreza,
        "nivel_logro_docente": nivel,
        "complejidad_destreza": complejidad,
        "carga_cognitiva": carga,
        "tipo_competencia": tipo
    }

    resultado = predecir_caso_personalizado(
        modelo_accion,
        modelo_estrategia,
        modelo_recurso,
        caso,
        nombre
    )

    # =========================
    # RESULTADO
    # =========================
    st.success("✅ Recomendación generada")
    st.text(resultado)

    import pandas as pd

    df_debug = pd.DataFrame([caso])
    st.write("📊 Input al modelo:")
    st.write(df_debug)

    proba = modelo_accion.predict_proba(df_debug)
    st.write("🔎 Probabilidades acción:")
    st.write(proba)

    proba = modelo_accion.predict_proba(df_debug)
    st.write(proba)