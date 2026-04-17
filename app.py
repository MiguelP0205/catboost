import streamlit as st
import pandas as pd
import io
#from utils import load_models_and_encoders, predecir_casos_personalizados_desde_df, generate_pdf_report, to_excel
import utils

st.set_page_config(layout="wide", page_title="Recomendador Pedagógico")

# --- Load Models and Encoders ---
@st.cache_resource
def get_models_and_encoders():
    return utils.load_models_and_encoders()

try:
    modelo_accion, modelo_estrategia, modelo_recurso, encoders, X_train_columns, X_train_rec_columns = get_models_and_encoders()
    st.success("Modelos y codificadores cargados exitosamente.")
except Exception as e:
    st.error(f"Error al cargar los modelos y encoders: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("Recomendador de Estrategias Educativas")
st.markdown("Esta aplicación predice la acción, estrategia y recurso pedagógico más adecuado para un grupo de estudiantes basado en sus datos de desempeño.")
st.markdown("---")

# Input form
st.header("1. Cargar Datos de Estudiantes")
st.markdown("Por favor, suba un archivo Excel con los datos de sus estudiantes. El archivo debe contener las siguientes columnas:")
st.markdown("**Columnas Requeridas:** `nombre_estudiante`, `curso`, `bloque`, `destreza`, `nivel_desempeno_estudiante`, `nivel_participacion_estudiante`, `complejidad_destreza`, `carga_cognitiva`, `tipo_competencia`, `tiempo_ensenanza`, `disponibilidad_tiempo`")
st.markdown("**Ejemplo de valores:**")
st.markdown("`curso`: '8vo', '9no', '10mo'")
st.markdown("`bloque`: 'Algebra', 'Geometria', 'Estadistica'")
st.markdown("`destreza`: 'Resolver ecuaciones lineales', 'Interpretar graficos', etc.")
st.markdown("`nivel_desempeno_estudiante`, `nivel_participacion_estudiante`: 0, 1, 2, 3 (0 si no asistió/no hay datos, 1 bajo, 2 medio, 3 alto)")
st.markdown("`complejidad_destreza`, `carga_cognitiva`: 1, 2, 3")
st.markdown("`tipo_competencia`: 'Conceptual', 'Procedimental', 'Resolucion_problemas'")
st.markdown("`tiempo_ensenanza`, `disponibilidad_tiempo`: Valores numéricos (horas)")

uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])

df_student_data = None
if uploaded_file is not None:
    try:
        df_student_data = pd.read_excel(uploaded_file)

        required_cols = [
            'nombre_estudiante', 'curso', 'bloque', 'destreza', 'nivel_desempeno_estudiante',
            'nivel_participacion_estudiante', 'complejidad_destreza', 'carga_cognitiva',
            'tipo_competencia', 'tiempo_ensenanza', 'disponibilidad_tiempo'
        ]

        if not all(col in df_student_data.columns for col in required_cols):
            st.error("El archivo Excel no contiene todas las columnas requeridas. Por favor, verifica el formato.")
            df_student_data = pd.DataFrame() # Clear dataframe if columns are missing

        st.success("Archivo cargado exitosamente!")
        st.dataframe(df_student_data.head()) # Vista previa de los primeros 5 estudiantes del curso en el excel

        # --- Make Predictions ---
        st.header("2. Resultados de Predicción")
        full_report, df_grades_for_download = utils.predecir_casos_personalizados_desde_df(
            modelo_accion, modelo_estrategia, modelo_recurso,
            df_student_data, encoders, X_train_columns, X_train_rec_columns
        )
        st.text(full_report)

        # --- Download Current Grades (Excel) ---
        st.header("3. Descargar Calificaciones")
        st.markdown("Puedes descargar una planilla con las calificaciones sugeridas para la destreza analizada.")
        st.download_button(
            label="Descargar Planilla de Calificaciones Actual",
            data=utils.to_excel(df_grades_for_download),
            file_name="planilla_calificaciones_prueba_actual.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- Download PDF Report ---
        st.subheader("4. Descargar Reporte en PDF")
        pdf_buffer = utils.generate_pdf_report(full_report)
        st.download_button(
            label="Descargar Reporte Pedagógico (PDF)",
            data=pdf_buffer.getvalue(),
            file_name="reporte_pedagogico.pdf",
            mime="application/pdf"
        )

        # --- Optional: Upload and Merge Previous Grades ---
        st.subheader("Opcional: Fusionar con Planilla de Calificaciones Existente")
        st.markdown("Si tienes una planilla de calificaciones previa (Excel con 'nombre_estudiante' y otras destrezas), puedes subirla y fusionarla con los resultados actuales.")
        uploaded_grades_file = st.file_uploader("Sube tu planilla de calificaciones previa", type=["xlsx"], key="previous_grades_uploader")

        if uploaded_grades_file is not None:
            try:
                df_previous_grades = pd.read_excel(uploaded_grades_file)
                st.success("Planilla de calificaciones previa cargada!")
                st.dataframe(df_previous_grades.head())

                # Merge logic
                # Assuming 'nombre_estudiante' is the common key
                if 'nombre_estudiante' in df_previous_grades.columns:
                    # Use outer merge to keep all students from both dataframes
                    df_merged_grades = pd.merge(df_previous_grades, df_grades_for_download, on='nombre_estudiante', how='outer')
                    st.subheader("Planilla de Calificaciones Fusionada")
                    st.dataframe(df_merged_grades)

                    st.download_button(
                        label="Descargar Planilla de Calificaciones Fusionada",
                        data=utils.to_excel(df_merged_grades),
                        file_name="nueva_planilla_calificaciones_completa.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("La planilla previa debe contener la columna 'nombre_estudiante' para poder fusionar.")
            except Exception as e_grades:
                st.error(f"Error al leer la planilla de calificaciones previa: {e_grades}")


    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")
else:
    st.info("Esperando que subas un archivo Excel para analizar.")