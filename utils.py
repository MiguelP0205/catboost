import pandas as pd

# =========================
# RANGO DE CALIFICACIÓN
# =========================
def sugerir_rango_calificacion(nivel):
    rangos = {
        1: "0 - 4",
        2: "4 - 6",
        3: "6 - 8",
        4: "8 - 9",
        5: "9 - 10"
    }
    return rangos.get(nivel, "N/A")


# =========================
# FUNCIÓN PRINCIPAL
# =========================
def predecir_caso_personalizado(
    modelo_accion,
    modelo_estrategia,
    modelo_recurso,
    caso,
    nombre_estudiante
):
    
    df_nuevo = pd.DataFrame([caso])
    
    pred_accion = modelo_accion.predict(df_nuevo)[0][0]
    pred_estrategia = modelo_estrategia.predict(df_nuevo)[0][0]
    
    df_nuevo_rec = df_nuevo.copy()
    df_nuevo_rec["estrategia"] = pred_estrategia
    
    pred_recurso = modelo_recurso.predict(df_nuevo_rec)[0][0]
    
    nivel = caso["nivel_logro_docente"]
    rango = sugerir_rango_calificacion(nivel)
    
    resultado = f"""
📘 Estudiante: {nombre_estudiante}

📚 Destreza evaluada: {caso['destreza']}

🔹 Acción docente recomendada: {pred_accion}
🔹 Estrategia sugerida: {pred_estrategia}
🔹 Recurso recomendado: {pred_recurso}

📊 Nivel de logro: {nivel}
🎯 Rango de calificación sugerido: {rango}, sobre 10

💡 Nota:
El docente puede asignar la calificación dentro de este rango 
según su criterio profesional.
"""
    
    return resultado