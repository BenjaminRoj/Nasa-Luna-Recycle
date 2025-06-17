import pandas as pd
import numpy as np

def load_mendeley_dataframe(file_path):
    """
    Carga el DataFrame real del dataset de Mendeley desde un archivo CSV.

    Args:
        file_path (str): La ruta al archivo CSV del dataset de Mendeley.

    Returns:
        pd.DataFrame: El DataFrame cargado, o None si hay un error.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"DataFrame cargado exitosamente desde '{file_path}'.")
        print("Primeras 5 filas:")
        print(df.head())
        print("\nColumnas del DataFrame:")
        print(df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta '{file_path}'.")
        return None
    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo CSV: {e}")
        return None

def calculate_average_composition_by_processing(df):
    """
    Calcula la composición química promedio (y el % de Al) para cada tipo de procesamiento
    y la suma total de impurezas/otros aleantes.

    Args:
        df (pd.DataFrame): DataFrame con el dataset de Mendeley.

    Returns:
        pd.DataFrame: DataFrame con la composición promedio por tipo de procesamiento.
    """
    if df is None:
        return None

    # Columnas de elementos químicos (ajustar si tu dataset tiene más o menos)
    # IMPORTANTE: Asegúrate de que esta lista coincida EXACTAMENTE con tus columnas de elementos en el CSV
    element_cols = ['Ag', 'Al', 'B', 'Be', 'Bi', 'Cd', 'Co', 'Cr', 'Cu', 'Er', 'Eu', 
                    'Fe', 'Ga', 'Li', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Si', 'Sn', 
                    'Ti', 'V', 'Zn', 'Zr'] 
    
    # Asegurarse de que las columnas de elementos existan en el DataFrame
    missing_cols = [col for col in element_cols if col not in df.columns]
    if missing_cols:
        print(f"Advertencia: Faltan las siguientes columnas de elementos en el DataFrame: {missing_cols}")
        print("Estas columnas no se incluirán en el cálculo de impurezas.")
        element_cols = [col for col in element_cols if col in df.columns]
    
    # Asegurarse de que 'Processing' exista
    if 'Processing' not in df.columns:
        print("Error: La columna 'Processing' no se encontró en el DataFrame.")
        return None

    # Convertir columnas de elementos a tipo numérico, manejando errores
    for col in element_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rellenar cualquier NaN que haya resultado de 'coerce' con 0, asumiendo que NaN significa ausencia
    df = df.fillna(0)

    # Calcular el porcentaje total de elementos que NO son Aluminio para cada fila
    # Esto incluye aleantes y otras impurezas.
    # Excluimos 'Al' de esta suma para que 'Total_Impurities' sea el resto.
    other_elements_cols = [col for col in element_cols if col != 'Al']
    df['Total_Impurities'] = df[other_elements_cols].sum(axis=1)

    # Agrupar por 'Processing' y calcular los promedios
    avg_composition = df.groupby('Processing')[['Al', 'Total_Impurities'] + other_elements_cols].mean()
    
    return avg_composition

def predict_al_output(
    cantidad_total_material_kg: float,
    porcentaje_estructural_struts: float,
    porcentaje_overwrap: float,
    porcentaje_drink_pouch: float,
    porcentaje_rehydratable_pouch: float,
    mendeley_avg_comp: pd.DataFrame,
    melting_efficiency: float = 0.95 # Factor de eficiencia de fundición (ej. 95%)
):
    """
    Calcula la cantidad de aluminio útil y desechos (impurezas) fundidos
    basándose en los tipos de materiales de entrada y su procesamiento asociado.

    Args:
        cantidad_total_material_kg (float): Cantidad total de material a fundir en kg.
        porcentaje_estructural_struts (float): Porcentaje del material total de 'Structural Struts'.
        porcentaje_overwrap (float): Porcentaje del material total de 'Overwrap'.
        porcentaje_drink_pouch (float): Porcentaje del material total de 'Drink Pouch'.
        porcentaje_rehydratable_pouch (float): Porcentaje del material total de 'Rehydratable Pouch'.
        mendeley_avg_comp (pd.DataFrame): DataFrame con la composición promedio por tipo de procesamiento.
        melting_efficiency (float): Eficiencia del proceso de fundición para el aluminio puro (ej. 0.95 = 95%).

    Returns:
        tuple: (kilos_desechos_totales, kilos_aluminio_util_total) en kg.
    """
    # Validar que los porcentajes suman aproximadamente 100%
    total_input_percentage = (porcentaje_estructural_struts + porcentaje_overwrap + 
                              porcentaje_drink_pouch + porcentaje_rehydratable_pouch)
    if not np.isclose(total_input_percentage, 100.0, atol=0.01):
        print(f"Advertencia: La suma de los porcentajes de entrada es {total_input_percentage:.2f}%, no 100%.")
        if total_input_percentage > 0:
            factor = 100.0 / total_input_percentage
            porcentaje_estructural_struts *= factor
            porcentaje_overwrap *= factor
            porcentaje_drink_pouch *= factor
            porcentaje_rehydratable_pouch *= factor
            print("Los porcentajes de entrada han sido normalizados.")
        else:
            print("Error: La suma de los porcentajes es 0. No hay material para procesar.")
            return 0, 0
    
    kilos_aluminio_util_total = 0.0
    kilos_desechos_total = 0.0 
    
    # Asignaciones de nuestros materiales a los tipos de Processing en Mendeley
    # Estas son las asignaciones que discutimos como las más plausibles.
    # Asegúrate de que los strings de los 'processing_type' coincidan exactamente con
    # los valores únicos de la columna 'Processing' en tu dataset de Mendeley.
    material_types = {
        "estructural_struts": {
            "percentage": porcentaje_estructural_struts,
            "processing_type": "Solutionised + Artificially peak aged" 
        },
        "overwrap": {
            "percentage": porcentaje_overwrap,
            "processing_type": "Strain hardened"
        },
        "drink_pouch": {
            "percentage": porcentaje_drink_pouch,
            "processing_type": "Strain hardened" # O "No Processing" si prefieres
        },
        "rehydratable_pouch": {
            "percentage": porcentaje_rehydratable_pouch,
            "processing_type": "Strain hardened" # O "No Processing" si prefieres
        }
    }

    print("\n--- Calculando por tipo de material ---")
    for material_name, data in material_types.items():
        percentage_input = data["percentage"] / 100.0
        processing_type = data["processing_type"]
        
        kilos_this_material = cantidad_total_material_kg * percentage_input
        
        if processing_type not in mendeley_avg_comp.index:
            print(f"Advertencia: El tipo de procesamiento '{processing_type}' no se encontró en los datos de Mendeley para '{material_name}'. Se asume 100% Al y 0% impurezas para este material.")
            # Si un tipo de procesamiento no está, se asume pureza máxima como fallback o se salta.
            # Aquí, para no detener el cálculo, asumo que es 100% Al si no se encuentra el procesamiento,
            # lo que puede ser demasiado optimista. Una alternativa sería omitirlo.
            al_purity_in_alloy = 1.0 # 100% Al
            impurities_in_alloy = 0.0 # 0% Impurezas
        else:
            composition = mendeley_avg_comp.loc[processing_type]
            al_purity_in_alloy = composition['Al'] / 100.0 
            impurities_in_alloy = composition['Total_Impurities'] / 100.0
        
        al_from_this_material = kilos_this_material * al_purity_in_alloy
        impurities_from_this_material = kilos_this_material * impurities_in_alloy
        
        al_useful_after_melting = al_from_this_material * melting_efficiency
        
        kilos_aluminio_util_total += al_useful_after_melting
        kilos_desechos_total += impurities_from_this_material 
        
        print(f"{material_name.replace('_', ' ').title()} ({percentage_input*100:.1f}% del total, {kilos_this_material:.2f} kg):")
        print(f"  Tipo de procesamiento: '{processing_type}'")
        print(f"  Aluminio en la aleación: {al_purity_in_alloy*100:.2f}%")
        print(f"  Impurezas/Otros en la aleación: {impurities_in_alloy*100:.2f}%")
        print(f"  Kilos de Al útil aportado (después de eficiencia): {al_useful_after_melting:.3f} kg")
        print(f"  Kilos de Desechos (impurezas) aportado: {impurities_from_this_material:.3f} kg")

    print("\n--- Resultados Finales ---")
    print(f"Eficiencia de Fundición Aplicada: {melting_efficiency*100:.1f}%")
    print(f"Cantidad Total de Material Ingresado: {cantidad_total_material_kg:.2f} kg")
    print(f"Cantidad Total de Aluminio Útil Fundido: {kilos_aluminio_util_total:.3f} kg")
    print(f"Cantidad Total de Desechos (Impurezas/Otros Aleantes): {kilos_desechos_total:.3f} kg")
    
    # Verificar la suma si es necesario
    total_output = kilos_aluminio_util_total + kilos_desechos_total
    print(f"Suma de Kilos de Aluminio Útil + Desechos: {total_output:.3f} kg (Nota: No igual al total de entrada debido a la eficiencia de fundición sobre el Al puro).")

    return kilos_desechos_total, kilos_aluminio_util_total

# --- PROCESO PRINCIPAL ---

if __name__ == "__main__":
    # ¡¡¡IMPORTANTE!!! REEMPLAZA 'tu_dataset_mendeley.csv' CON LA RUTA Y NOMBRE REAL DE TU ARCHIVO CSV
    file_path_mendeley = 'C:/Users/benja/Downloads/al_data.csv' 
    
    # 1. Cargar el DataFrame de Mendeley
    mendeley_df_raw = load_mendeley_dataframe(file_path_mendeley)

    if mendeley_df_raw is None:
        print("No se pudo cargar el DataFrame de Mendeley. Saliendo del programa.")
    else:
        # 2. Calcular la composición promedio por tipo de procesamiento
        mendeley_avg_composition = calculate_average_composition_by_processing(mendeley_df_raw.copy()) # Usar una copia para evitar SettingWithCopyWarning
        
        if mendeley_avg_composition is None:
            print("No se pudo calcular la composición promedio. Saliendo del programa.")
        else:
            print("\nComposición Promedio por Tipo de Procesamiento (solo Al y Total_Impurities):")
            print(mendeley_avg_composition[['Al', 'Total_Impurities']].round(3))
            print("\n--- Modelo de Predicción de Output de Fundición ---")

            # --- PARÁMETROS DE ENTRADA DEL USUARIO ---
            # Puedes cambiar estos valores para probar diferentes escenarios
            total_material_input_kg = 100.0 # Cantidad total de material a ingresar al horno (en kg)
            
            # Los 4 porcentajes correspondientes a la cantidad de cada "desecho" (elemento estructural)
            # Asegúrate de que la suma de estos porcentajes sea 100%
            pct_estructural_struts = 25.0 
            pct_overwrap = 25.0    
            pct_drink_pouch = 25.0 
            pct_rehydratable_pouch = 25.0 
            
            # Ejecutar el modelo con los parámetros definidos
            kilos_desechos, kilos_aluminio_util = predict_al_output(
                cantidad_total_material_kg=total_material_input_kg,
                porcentaje_estructural_struts=pct_estructural_struts,
                porcentaje_overwrap=pct_overwrap,
                porcentaje_drink_pouch=pct_drink_pouch,
                porcentaje_rehydratable_pouch=pct_rehydratable_pouch,
                mendeley_avg_comp=mendeley_avg_composition
            )
            print(f"\n--- Resumen de Predicción ---")
            print(f"Para {total_material_input_kg:.2f} kg de material ingresado:")
            print(f"  Se esperan {kilos_aluminio_util:.3f} kg de aluminio útil fundido.")
            print(f"  Se esperan {kilos_desechos:.3f} kg de desechos (impurezas/otros aleantes).")
