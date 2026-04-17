import pandas as pd
import random
import joblib
from collections import Counter

# Imports for PDF generation
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.units import inch

# --- Global definitions (from Colab notebook) ---
destreza_map = {
    # ---------- Resolución ----------
    "Resolver ecuaciones lineales": "Resolucion",
    "Resolver ecuaciones cuadraticas": "Resolucion",
    "Resolver sistemas de ecuaciones": "Resolucion",

    # ---------- Procedimiento ----------
    "Operar con fracciones": "Procedimiento",
    "Simplificar expresiones": "Procedimiento",
    "Aplicar algoritmos": "Procedimiento",

    # ---------- Aplicación ----------
    "Calcular areas": "Aplicacion",
    "Aplicar proporcionalidad": "Aplicacion",
    "Resolver problemas geometricos": "Aplicacion",

    # ---------- Interpretación ----------
    "Interpretar graficos": "Interpretacion",
    "Analizar datos": "Interpretacion",
    "Leer tablas": "Interpretacion",

    # ---------- Modelación ----------
    "Plantear ecuaciones": "Modelacion",
    "Traducir problemas a lenguaje matematico": "Modelacion",

    # ---------- Razonamiento ----------
    "Justificar procedimientos": "Razonamiento",
    "Argumentar soluciones": "Razonamiento"
}

categorical_cols_for_encoding = [
    "curso", "bloque", "destreza", "tipo_competencia",
    "accion_docente", "estrategia", "recurso", "nivel_logro_docente"
]

# --- Helper Functions ---
def sugerir_rango_calificacion(nivel):
    if nivel == 1:
        return "1 - 6"
    elif nivel == 2:
        return "7 - 8"
    elif nivel == 3:
        return "9 - 10"

def calcular_nivel_final_estudiante(nivel_desempeno, nivel_participacion):
    if nivel_desempeno == 0 and nivel_participacion == 0:
        return 1
    average_nivel = (nivel_desempeno + nivel_participacion) / 2
    if average_nivel <= 1.5:
        return 1
    elif average_nivel <= 2.5:
        return 2
    else:
        return 3

def map_average_to_level(average_score):
    if average_score <= 1.5:
        return 1
    elif average_score <= 2.5:
        return 2
    else:
        return 3

# --- Model Loading Function ---
def load_models_and_encoders():
    try:
        modelo_accion = joblib.load("models/modelo_accion_rf.joblib")
        modelo_estrategia = joblib.load("models/modelo_estrategia_rf.joblib")
        modelo_recurso = joblib.load("models/modelo_recurso_rf.joblib")
        encoders = joblib.load("models_requirements/encoders.joblib")
        X_train_columns = joblib.load("models_requirements/X_train_columns.joblib")
        X_train_rec_columns = joblib.load("models_requirements/X_train_rec_columns.joblib")
        return modelo_accion, modelo_estrategia, modelo_recurso, encoders, X_train_columns, X_train_rec_columns
    except FileNotFoundError as e:
        raise Exception(f"Error loading model files. Ensure all .joblib files are in the current directory: {e}")

def predecir_casos_personalizados_desde_df(
    modelo_accion,
    modelo_estrategia,
    modelo_recurso,
    df_casos_nuevos,
    encoders,
    X_train_columns,
    X_train_rec_columns
):
    total_estudiantes_grupo = len(df_casos_nuevos)
    if total_estudiantes_grupo == 0:
        return "No hay estudiantes para analizar.", pd.DataFrame()

    count_nivel_1 = ((df_casos_nuevos['nivel_desempeno_estudiante'] == 1) | (df_casos_nuevos['nivel_desempeno_estudiante'] == 0)).sum()
    count_nivel_2 = (df_casos_nuevos['nivel_desempeno_estudiante'] == 2).sum()
    count_nivel_3 = (df_casos_nuevos['nivel_desempeno_estudiante'] == 3).sum()

    grupo_pct_bajo = round((count_nivel_1 / total_estudiantes_grupo) * 100, 2)
    grupo_pct_medio = round((count_nivel_2 / total_estudiantes_grupo) * 100, 2)
    grupo_pct_alto = round((count_nivel_3 / total_estudiantes_grupo) * 100, 2)

    cols_to_encode_for_prediction = [
        "curso", "bloque", "destreza", "tipo_competencia"
    ]

    resultados_individuales = []
    predicciones_accion_encoded_list = []
    predicciones_estrategia_encoded_list = []
    predicciones_recurso_encoded_list = []
    grades_data = []

    first_student_case = df_casos_nuevos.iloc[0].drop('nombre_estudiante').to_dict()
    curso_general = first_student_case['curso']
    bloque_general = first_student_case['bloque']
    destreza_general = first_student_case['destreza']
    tipo_competencia_general = first_student_case['tipo_competencia']
    complejidad_general = first_student_case['complejidad_destreza']
    carga_general = first_student_case['carga_cognitiva']
    tiempo_ensenanza_general = first_student_case['tiempo_ensenanza']
    disponibilidad_tiempo_general = first_student_case['disponibilidad_tiempo']

    destreza_categoria_general = destreza_map.get(destreza_general, "No clasificada")

    avg_nivel_desempeno_grupo = df_casos_nuevos['nivel_desempeno_estudiante'].mean()
    avg_nivel_participacion_grupo = df_casos_nuevos['nivel_participacion_estudiante'].mean()

    pred_nivel_logro_docente_general = map_average_to_level(avg_nivel_desempeno_grupo)

    for idx, student_data_series in df_casos_nuevos.iterrows():
        nombre_estudiante = student_data_series['nombre_estudiante']
        caso = student_data_series.drop('nombre_estudiante').to_dict()

        df_prediccion_features = student_data_series.drop('nombre_estudiante').to_frame().T

        df_prediccion_features['pct_bajo'] = grupo_pct_bajo
        df_prediccion_features['pct_medio'] = grupo_pct_medio
        df_prediccion_features['pct_alto'] = grupo_pct_alto

        df_nuevo = df_prediccion_features[X_train_columns].copy()

        for col in cols_to_encode_for_prediction:
            if col in df_nuevo.columns and col in encoders:
                df_nuevo[col] = encoders[col].transform(df_nuevo[col])

        pred_accion_encoded = modelo_accion.predict(df_nuevo).item()
        pred_estrategia_encoded = modelo_estrategia.predict(df_nuevo).item()

        pred_accion = encoders['accion_docente'].inverse_transform([pred_accion_encoded]).item()
        pred_estrategia = encoders['estrategia'].inverse_transform([pred_estrategia_encoded]).item()

        df_nuevo_rec = df_nuevo.copy()
        df_nuevo_rec["estrategia"] = pred_estrategia_encoded

        df_nuevo_rec = df_nuevo_rec[X_train_rec_columns]

        pred_recurso_encoded = modelo_recurso.predict(df_nuevo_rec).item()
        pred_recurso = encoders['recurso'].inverse_transform([pred_recurso_encoded]).item()

        predicciones_accion_encoded_list.append(pred_accion_encoded)
        predicciones_estrategia_encoded_list.append(pred_estrategia_encoded)
        predicciones_recurso_encoded_list.append(pred_recurso_encoded)

        nivel_desempeno_estudiante_caso = caso['nivel_desempeno_estudiante']
        nivel_participacion_estudiante_caso = caso['nivel_participacion_estudiante']

        nivel_final_estudiante = calcular_nivel_final_estudiante(nivel_desempeno_estudiante_caso, nivel_participacion_estudiante_caso)
        rango_calificacion_sugerido = sugerir_rango_calificacion(nivel_final_estudiante)

        grades_data.append({
            'nombre_estudiante': nombre_estudiante,
            destreza_general: rango_calificacion_sugerido
        })

        explicacion_accion = f"La acción '{pred_accion}' se recomienda debido a que el nivel de logro actual del estudiante es Nivel {nivel_desempeno_estudiante_caso}. Considerando la complejidad ({complejidad_general}/3) y la carga cognitiva ({carga_general}/3) de la destreza. "
        if pred_accion == 'reenseñar':
            explicacion_accion += "Este nivel sugiere que el estudiante necesita una revisión fundamental de los conceptos básicos para construir una base sólida. Es esencial enfocarse en los prerrequisitos y aclarar cualquier malentendido inicial."
        elif pred_accion == 'reforzar':
            explicacion_accion += "El estudiante muestra un entendimiento parcial, por lo que es vital consolidar el conocimiento y abordar áreas específicas de debilidad. Un refuerzo estratégico ayudará a cerrar las brechas antes de progresar."
        elif pred_accion == 'retroalimentar':
            explicacion_accion += "El desempeño del estudiante es bueno, indicando una comprensión sólida. La retroalimentación detallada y específica le permitirá refinar su conocimiento, corregir errores menores y profundizar en la aplicación de la destreza."
        else:
            explicacion_accion += "El estudiante demuestra un dominio claro de la destreza. Para mantener el compromiso y estimular el crecimiento, se le puede desafiar con aplicaciones más avanzadas o nuevos contextos que requieran esta destreza."

        explicacion_estrategia = f"La estrategia '{pred_estrategia}' se sugiere como la más efectiva para el nivel actual del estudiante y el tipo de competencia '{tipo_competencia_general}'. "
        if pred_estrategia == 'tradicional':
            explicacion_estrategia += "Este enfoque es eficaz para la transmisión directa de conocimientos, especialmente en competencias conceptuales donde la claridad y estructura son clave para el aprendizaje inicial."
        elif pred_estrategia == 'colaborativa':
            explicacion_estrategia += "Promueve la interacción entre pares, el intercambio de ideas y la construcción conjunta del conocimiento. Es ideal para reforzar habilidades procedimentales y desarrollar el pensamiento crítico en un entorno de apoyo."
        else:
            explicacion_estrategia += "Esta estrategia fomenta la aplicación práctica del conocimiento para resolver desafíos del mundo real. Es óptima para competencias de resolución de problemas y para desarrollar habilidades de análisis y síntesis en un nivel avanzado."

        explicacion_recurso = f"Para complementar la estrategia '{pred_estrategia}' y facilitar el aprendizaje, se recomienda el recurso '{pred_recurso}'. "
        if pred_recurso == 'Libro':
            explicacion_recurso += "Proporciona una base teórica sólida y ejercicios estructurados, ideal para el estudio independiente o la referencia."
        elif pred_recurso == 'Guia':
            explicacion_recurso += "Ofrece actividades prácticas y paso a paso, muy útil para el desarrollo de habilidades procedimentales y el trabajo en grupo."
        elif pred_recurso == 'TIC':
            explicacion_recurso += "Permite el acceso a herramientas interactivas, simulaciones y recursos multimedia, enriqueciendo la experiencia de aprendizaje y adaptándose a diversos estilos."
        else:
            explicacion_recurso += "Facilita la exploración interactiva, la visualización de conceptos complejos y la resolución de problemas avanzados mediante la computación. Ideal para potenciar la comprensión y la experimentación."

        resultado_individual = f"""
📘 Estudiante: {nombre_estudiante}

--- Análisis del Caso ---
*   Nivel de Desempeño del Estudiante: {nivel_desempeno_estudiante_caso}
*   Nivel de Participación del Estudiante: {nivel_participacion_estudiante_caso}

--- Recomendaciones ---
🔹 **Acción docente recomendada:** {pred_accion}
    *   Explicación: {explicacion_accion}
🔹 **Estrategia sugerida:** {pred_estrategia}
    *   Explicación: {explicacion_estrategia}
🔹 **Recurso recomendado:** {pred_recurso}
    *   Explicación: {explicacion_recurso}

--- Calificación Sugerida ---
📊 Nivel de Logro Final del Estudiante: {nivel_final_estudiante}
🎯 Rango de calificación sugerido: {rango_calificacion_sugerido}, sobre 10
"""
        resultados_individuales.append(resultado_individual)

    df_grades_for_download = pd.DataFrame(grades_data)

    if predicciones_accion_encoded_list:
        predicciones_accion_decoded = encoders['accion_docente'].inverse_transform(predicciones_accion_encoded_list)
        predicciones_estrategia_decoded = encoders['estrategia'].inverse_transform(predicciones_estrategia_encoded_list)
        predicciones_recurso_decoded = encoders['recurso'].inverse_transform(predicciones_recurso_encoded_list)

        accion_mas_frecuente = Counter(predicciones_accion_decoded).most_common(1)[0][0]
        estrategia_mas_frecuente = Counter(predicciones_estrategia_decoded).most_common(1)[0][0]
        recurso_mas_frecuente = Counter(predicciones_recurso_decoded).most_common(1)[0][0]

        explicacion_accion_general = ""
        if accion_mas_frecuente == 'reenseñar':
            explicacion_accion_general = "La mayoría de los estudiantes requieren una intervención enfocada en los fundamentos. Se sugiere una revisión profunda de los conceptos básicos para asegurar una comprensión sólida."
        elif accion_mas_frecuente == 'reforzar':
            explicacion_accion_general = "Una parte significativa del grupo se beneficiaría de actividades de refuerzo. Es clave consolidar los conocimientos existentes y resolver dudas específicas para cerrar brechas."
        elif accion_mas_frecuente == 'retroalimentar':
            explicacion_accion_general = "El grupo en general demuestra un buen nivel de comprensión. La retroalimentación constructiva será esencial para refinar sus habilidades y profundizar en la aplicación de la destreza."
        else:
            explicacion_accion_general = "La mayoría de los estudiantes han dominado la destreza. Se recomienda presentarles desafíos más complejos y nuevos contextos de aplicación para estimular su crecimiento y mantener el interés."

        explicacion_estrategia_general = ""
        if estrategia_mas_frecuente == 'tradicional':
            explicacion_estrategia_general = "Para este grupo, un enfoque tradicional es óptimo. Esto implica la transmisión directa de conocimientos, explicaciones claras y ejercicios estructurados para asegurar la comprensión de los conceptos."
        elif estrategia_mas_frecuente == 'colaborativa':
            explicacion_estrategia_general = "La estrategia colaborativa es la más adecuada para el curso. Fomenta el trabajo en equipo, el intercambio de ideas y la construcción conjunta del aprendizaje, ideal para desarrollar habilidades y resolver problemas en conjunto."
        else:
            explicacion_estrategia_general = "Se recomienda una estrategia basada en problemas para el curso. Esto desafiará a los estudiantes a aplicar sus conocimientos en situaciones reales, desarrollando su pensamiento crítico y capacidad de resolución."

        explicacion_recurso_general = ""
        if recurso_mas_frecuente == 'Libro':
            explicacion_recurso_general = "El uso predominante de libros de texto y materiales impresos proporcionará una base teórica sólida y ejercicios prácticos, adecuados para el estudio individual y la referencia en clase."
        elif recurso_mas_frecuente == 'Guia':
            explicacion_recurso_general = "Las guías didácticas son el recurso más sugerido. Ofrecen actividades prácticas, ejemplos y ejercicios paso a paso, facilitando el desarrollo de habilidades y la aplicación de conceptos."
        elif recurso_mas_frecuente == 'TIC':
            explicacion_recurso_general = "La implementación de Tecnologías de la Información y Comunicación (TIC) enriquecerá el aprendizaje. Herramientas interactivas, simulaciones y recursos multimedia captarán la atención y facilitarán la comprensión de temas complejos."
        else:
            explicacion_recurso_general = "El software educativo es el recurso más apropiado para este grupo. Permite la exploración interactiva, la visualización de datos, la simulación de procesos y la resolución de problemas avanzados, fomentando un aprendizaje más dinámico y aplicado."

    else:
        accion_mas_frecuente = "N/A"
        estrategia_mas_frecuente = "N/A"
        recurso_mas_frecuente = "N/A"
        explicacion_accion_general = ""
        explicacion_estrategia_general = ""
        explicacion_recurso_general = ""

    resumen_curso = f"""
--- REPORTE GENERAL DEL CURSO ---

**Información General del Curso:**
*   Curso: {curso_general}
*   Bloque: {bloque_general}
*   Destreza: {destreza_general}
*   Tipo de Competencia: {tipo_competencia_general}
*   Complejidad de la Destreza: {complejidad_general}/3
*   Carga Cognitiva: {carga_general}/3
*   Tiempo estimado de enseñanza: {tiempo_ensenanza_general} horas
*   Disponibilidad de tiempo restante en el bloque/trimestre: {disponibilidad_tiempo_general} horas
*   Nivel de Logro Docente Predicho para el Curso: {pred_nivel_logro_docente_general}

**Análisis de Desempeño del Grupo:**
*   Estudiantes con desempeño bajo: {grupo_pct_bajo}%
*   Estudiantes con desempeño medio: {grupo_pct_medio}%
*   Estudiantes con desempeño alto: {grupo_pct_alto}%

**Recomendaciones Generales para el Curso:**
*   **Acción docente predominante:** {accion_mas_frecuente}
    *   Explicación: {explicacion_accion_general}
*   **Estrategia más sugerida:** {estrategia_mas_frecuente}
    *   Explicación: {explicacion_estrategia_general}
*   **Recurso más recomendado:** {recurso_mas_frecuente}
    *   Explicación: {explicacion_recurso_general}

"""

    full_report_text = resumen_curso + "\n" + "\n-- DETALLE INDIVIDUAL DE ESTUDIANTES --\n\n" + "\n".join(resultados_individuales)
    return full_report_text, df_grades_for_download

def generate_pdf_report(report_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch,
                            leftMargin=inch,
                            topMargin=inch,
                            bottomMargin=inch)
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='ReportContent', alignment=TA_JUSTIFY, fontSize=10, leading=12))
    styles.add(ParagraphStyle(name='SeparatorLine', alignment=TA_JUSTIFY, fontSize=8, leading=10, textColor='#A0A0A0'))

    story = []

    story.append(Paragraph("<b>Reporte Pedagógico Detallado</b>", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    for line in report_text.split('\n'):
        if line.strip().startswith('---'):
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(f"<b>{line.strip().replace('---', '').strip()}</b>", styles['h2']))
            story.append(Spacer(1, 0.1 * inch))
        elif line.strip().startswith('📘 Estudiante:'):
            if len(story) > 0 and 'DETALLE INDIVIDUAL DE ESTUDIANTES' in report_text and report_text.find('📘 Estudiante:') < report_text.find(line):
                story.append(Spacer(1, 0.3 * inch))
                story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", styles['SeparatorLine']))
                story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(f"<b>{line.strip()}</b>", styles['ReportContent']))
        elif line.strip().startswith('*'):
            story.append(Paragraph(line.strip(), styles['ReportContent']))
        elif line.strip().startswith('🔹'):
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(f"<b>{line.strip()}</b>", styles['ReportContent']))
        elif line.strip():
            story.append(Paragraph(line.strip(), styles['ReportContent']))
        else:
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    processed_data = output.getvalue()
    return processed_data