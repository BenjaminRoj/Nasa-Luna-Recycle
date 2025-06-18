import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def load_mendeley_dataframe(file_path):
    """
    Carga el DataFrame real del dataset de Mendeley desde un archivo CSV.
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
    """
    if df is None:
        return None

    # Columnas de elementos químicos relevantes para la composición.
    # AJUSTA ESTA LISTA para que coincida EXACTAMENTE con las columnas de tu CSV
    element_cols = ['Ag', 'Al', 'B', 'Be', 'Bi', 'Cd', 'Co', 'Cr', 'Cu', 'Er', 'Eu', 
                    'Fe', 'Ga', 'Li', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Si', 'Sn', 
                    'Ti', 'V', 'Zn', 'Zr'] 
    
    missing_elements = [col for col in element_cols if col not in df.columns]
    if missing_elements:
        print(f"Advertencia: Faltan las siguientes columnas de elementos en el DataFrame: {missing_elements}.")
        print("Estas columnas no se incluirán en el cálculo de impurezas y podrían afectar la precisión.")
        element_cols = [col for col in element_cols if col in df.columns] 
    
    if 'Processing' not in df.columns:
        print("Error: La columna 'Processing' no se encontró en el DataFrame.")
        return None

    for col in element_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0)

    other_elements_cols = [col for col in element_cols if col != 'Al']
    
    if 'Al' not in element_cols:
        print("Error: La columna 'Al' (Aluminio) no se encontró en la lista de elementos. No se puede calcular la pureza.")
        return None

    df['Total_Impurities'] = df[other_elements_cols].sum(axis=1)
    
    df['Sum_Elements'] = df[element_cols].sum(axis=1)
    df['Sum_Elements'] = df['Sum_Elements'].replace(0, 1) # Evitar división por cero

    for col in element_cols:
        df[col] = (df[col] / df['Sum_Elements']) * 100.0 # Normalizar a porcentaje (%)

    df['Total_Impurities'] = (df['Total_Impurities'] / df['Sum_Elements']) * 100.0 # Recalcular impurezas normalizadas

    avg_composition = df.groupby('Processing')[['Al', 'Total_Impurities'] + other_elements_cols].mean()
    
    return avg_composition

def generate_synthetic_data(mendeley_avg_comp: pd.DataFrame, n_samples: int = 5000, 
                            fixed_total_kg: float = 100.0, melting_efficiency: float = 0.95,
                            noise_level: float = 0.01): # Nuevo parámetro para el nivel de ruido
    """
    Genera un dataset sintético para entrenar el modelo de ML.
    Cada muestra es una combinación aleatoria de porcentajes de los 4 tipos de materiales.
    Los outputs se calculan usando la lógica determinística anterior, con ruido añadido.
    """
    
    X = [] # Características de entrada (porcentajes de cada tipo de material)
    Y_al_useful = [] # Salida: kilos de aluminio útil
    Y_impurities = [] # Salida: kilos de desechos (impurezas)
    Y_other_residues = [] # Salida: kilos de otros residuos/pérdidas

    # Mapeo de nuestros materiales a los tipos de Processing en Mendeley
    processing_assignments = {
        "estructural_struts": "Solutionised  + Artificially peak aged", 
        "overwrap": "Solutionised  + Artificially peak aged",
        "drink_pouch": "Strain hardened", 
        "rehydratable_pouch": "Strain hardened" 
    }
    
    available_processing_data = {
        name: mendeley_avg_comp.loc[proc_type] 
        for name, proc_type in processing_assignments.items() 
        if proc_type in mendeley_avg_comp.index
    }

    if not available_processing_data:
        print("Error: Ninguno de los tipos de procesamiento asignados se encontró en el dataset de Mendeley.")
        return None, None

    for _ in range(n_samples):
        # Generar porcentajes aleatorios que sumen 100%
        random_pcts = np.random.rand(4)
        random_pcts = random_pcts / random_pcts.sum() * 100.0 # Normalizar a 100%

        pct_estructural_struts = random_pcts[0]
        pct_overwrap = random_pcts[1]
        pct_drink_pouch = random_pcts[2]
        pct_rehydratable_pouch = random_pcts[3]

        # Simular el cálculo de salida usando la lógica determinística (nuestra "verdad")
        current_al_useful = 0.0
        current_impurities = 0.0
        
        material_data_for_sim = {
            "estructural_struts": {"percentage_input": pct_estructural_struts, "processing_type": processing_assignments["estructural_struts"]},
            "overwrap": {"percentage_input": pct_overwrap, "processing_type": processing_assignments["overwrap"]},
            "drink_pouch": {"percentage_input": pct_drink_pouch, "processing_type": processing_assignments["drink_pouch"]},
            "rehydratable_pouch": {"percentage_input": pct_rehydratable_pouch, "processing_type": processing_assignments["rehydratable_pouch"]}
        }

        for material_name, data in material_data_for_sim.items():
            percentage_input = data["percentage_input"] / 100.0
            processing_type = data["processing_type"]
            
            if processing_type in mendeley_avg_comp.index:
                kilos_this_material = fixed_total_kg * percentage_input
                composition = mendeley_avg_comp.loc[processing_type]
                
                al_purity_in_alloy = composition['Al'] / 100.0 
                impurities_in_alloy = composition['Total_Impurities'] / 100.0
                
                al_from_this_material = kilos_this_material * al_purity_in_alloy
                impurities_from_this_material = kilos_this_material * impurities_in_alloy
                
                al_useful_after_melting = al_from_this_material * melting_efficiency
                
                current_al_useful += al_useful_after_melting
                current_impurities += impurities_from_this_material
        
        current_total_output_calculated = current_al_useful + current_impurities
        current_other_residues = fixed_total_kg - current_total_output_calculated

        # --- AÑADIR RUIDO GAUSSIANO ---
        # Se añade ruido proporcional al fixed_total_kg para que sea significativo pero no dominante
        # El ruido se distribuye entre las tres categorías de salida para mantener la suma total.
        noise_al = np.random.normal(0, noise_level * fixed_total_kg)
        noise_imp = np.random.normal(0, noise_level * fixed_total_kg)
        noise_other = np.random.normal(0, noise_level * fixed_total_kg)
        
        # Ajustar para que la suma del ruido sea cero y no altere el total fixed_total_kg
        total_noise = noise_al + noise_imp + noise_other
        noise_al -= total_noise / 3
        noise_imp -= total_noise / 3
        noise_other -= total_noise / 3


        current_al_useful_noisy = max(0, current_al_useful + noise_al)
        current_impurities_noisy = max(0, current_impurities + noise_imp)
        current_other_residues_noisy = max(0, current_other_residues + noise_other)

        # Después de añadir el ruido, renormalizar para que la suma total siga siendo fixed_total_kg
        sum_noisy_outputs = current_al_useful_noisy + current_impurities_noisy + current_other_residues_noisy
        if sum_noisy_outputs > 0: # Evitar división por cero
            factor_re_normalize = fixed_total_kg / sum_noisy_outputs
            current_al_useful_noisy *= factor_re_normalize
            current_impurities_noisy *= factor_re_normalize
            current_other_residues_noisy *= factor_re_normalize
        else: # Si todos los ruidos y salidas originales eran 0, mantenerlas en 0
            current_al_useful_noisy = 0
            current_impurities_noisy = 0
            current_other_residues_noisy = 0


        # Añadir a las listas de entrenamiento
        X.append([pct_estructural_struts, pct_overwrap, pct_drink_pouch, pct_rehydratable_pouch])
        Y_al_useful.append(current_al_useful_noisy)
        Y_impurities.append(current_impurities_noisy)
        Y_other_residues.append(current_other_residues_noisy)
    
    # Convertir a arrays de numpy
    X = np.array(X)
    Y = np.column_stack((Y_al_useful, Y_impurities, Y_other_residues))
    
    print(f"\nSe generaron {n_samples} muestras de datos sintéticos con ruido (nivel: {noise_level*100:.1f}% de {fixed_total_kg} kg) para entrenamiento.")
    print(f"Forma de X (Características): {X.shape}")
    print(f"Forma de Y (Objetivos: Al útil, Impurezas, Otros Residuos): {Y.shape}")
    return X, Y

def train_ml_models(X_train, Y_train):
    """
    Entrena tres modelos de Regresión Lineal, uno para cada tipo de salida.
    """
    print("\n--- Entrenamiento de Modelos de Machine Learning ---")
    
    # Separar los objetivos
    Y_train_al_useful = Y_train[:, 0]
    Y_train_impurities = Y_train[:, 1]
    Y_train_other_residues = Y_train[:, 2]

    # Entrenar un modelo para cada objetivo
    model_al_useful = LinearRegression()
    model_impurities = LinearRegression()
    model_other_residues = LinearRegression()

    model_al_useful.fit(X_train, Y_train_al_useful)
    model_impurities.fit(X_train, Y_train_impurities)
    model_other_residues.fit(X_train, Y_train_other_residues)
    
    print("Modelos de Regresión Lineal entrenados.")
    return model_al_useful, model_impurities, model_other_residues

def evaluate_models(models, X_test, Y_test):
    """
    Evalúa el rendimiento de los modelos entrenados en el conjunto de prueba.
    """
    print("\n--- Evaluación de Modelos ---")
    metrics = {}
    
    output_names = ["Aluminio Útil", "Desechos (Impurezas)", "Otros Residuos/Pérdidas"]
    
    for i, model in enumerate(models):
        y_pred = model.predict(X_test)
        r2 = r2_score(Y_test[:, i], y_pred)
        mse = mean_squared_error(Y_test[:, i], y_pred)
        
        print(f"Modelo para {output_names[i]}:")
        print(f"  R-cuadrado (R2): {r2:.4f}")
        print(f"  Error Cuadrático Medio (MSE): {mse:.4f}")
        metrics[output_names[i]] = {"R2": r2, "MSE": mse}
    return metrics

def predict_al_output_with_ml(
    cantidad_total_material_kg: float,
    porcentaje_estructural_struts: float,
    porcentaje_overwrap: float,
    porcentaje_drink_pouch: float,
    porcentaje_rehydratable_pouch: float,
    trained_models: tuple,
    training_unit_kg: float # Necesitamos saber con qué unidad se entrenó
):
    """
    Predice la cantidad de aluminio útil, desechos y otros residuos utilizando los modelos ML entrenados.
    """
    model_al_useful, model_impurities, model_other_residues = trained_models

    input_pcts = np.array([porcentaje_estructural_struts, porcentaje_overwrap, 
                           porcentaje_drink_pouch, porcentaje_rehydratable_pouch])
    
    total_input_percentage = input_pcts.sum()
    if not np.isclose(total_input_percentage, 100.0, atol=0.01):
        print(f"Advertencia: La suma de los porcentajes de entrada es {total_input_percentage:.2f}%, no 100%. Se normalizarán para la predicción.")
        if total_input_percentage > 0:
            input_pcts = (input_pcts / total_input_percentage) * 100.0
        else:
            print("Error: La suma de los porcentajes es 0. No se puede predecir.")
            return 0, 0, 0
    
    X_predict = input_pcts.reshape(1, -1) # [[pct1, pct2, pct3, pct4]]

    # Realizar predicciones para la unidad de peso usada en el entrenamiento (ej. 100 kg)
    predicted_al_useful_per_unit = model_al_useful.predict(X_predict)[0]
    predicted_impurities_per_unit = model_impurities.predict(X_predict)[0]
    predicted_other_residues_per_unit = model_other_residues.predict(X_predict)[0]

    # Escalar las predicciones al total de kilos de entrada del usuario
    # Multiplicamos por (cantidad_total_material_kg / training_unit_kg) para escalar correctamente
    scaling_factor = cantidad_total_material_kg / training_unit_kg

    kilos_aluminio_util = predicted_al_useful_per_unit * scaling_factor
    kilos_desechos_impurezas = predicted_impurities_per_unit * scaling_factor
    kilos_otros_residuos = predicted_other_residues_per_unit * scaling_factor
    
    # Asegurarse de que no haya negativos pequeños debido a imprecisiones del modelo lineal o ruido
    kilos_aluminio_util = max(0, kilos_aluminio_util)
    kilos_desechos_impurezas = max(0, kilos_desechos_impurezas)
    kilos_otros_residuos = max(0, kilos_otros_residuos)

    print("\n--- Resultados de Predicción del Modelo ML ---")
    print(f"Cantidad Total de Material Ingresado: {cantidad_total_material_kg:.2f} kg")
    print(f"Cantidad Total de Aluminio Útil Fundido: {kilos_aluminio_util:.3f} kg")
    print(f"Cantidad Total de Desechos (Impurezas/Otros Aleantes Fundidos): {kilos_desechos_impurezas:.3f} kg")
    print(f"Cantidad de Otros Residuos/Pérdidas (No Al útil ni Impurezas fundidas): {kilos_otros_residuos:.3f} kg")
    
    total_output_sum_check = kilos_aluminio_util + kilos_desechos_impurezas + kilos_otros_residuos
    print(f"Suma Total de Outputs: {total_output_sum_check:.3f} kg. (Debería ser ~igual al total ingresado)")

    return kilos_desechos_impurezas, kilos_aluminio_util, kilos_otros_residuos

# --- PROCESO PRINCIPAL ---

if __name__ == "__main__":
    # --- Configuración ---
    # ¡¡¡IMPORTANTE!!! REEMPLAZA 'tu_dataset_mendeley.csv' CON LA RUTA Y NOMBRE REAL DE TU ARCHIVO CSV
    file_path_mendeley = 'C:/Users/benja/Downloads/al_data.csv' 
    N_SAMPLES_SYNTHETIC = 10000 # Número de muestras sintéticas para entrenar el modelo ML
    TRAINING_UNIT_KG = 100.0 # Cantidad base de kg para la cual se generarán los datos de entrenamiento (ej. 100 kg)
    # Nivel de ruido: 0.01 significa que el ruido tendrá una desviación estándar del 1% de los fixed_total_kg
    # Ajusta este valor para controlar cuán "realista" (ruidoso) quieres que sea el entrenamiento.
    # Un valor más alto simula más variabilidad en el proceso de fundición.
    NOISE_LEVEL = 0.005 # Por ejemplo, 0.5% de desviación estándar del total_kg

    # 1. Cargar el DataFrame de Mendeley
    mendeley_df_raw = load_mendeley_dataframe(file_path_mendeley)

    if mendeley_df_raw is None:
        print("No se pudo cargar el DataFrame de Mendeley. Saliendo del programa.")
    else:
        # 2. Calcular la composición promedio por tipo de procesamiento
        mendeley_avg_composition = calculate_average_composition_by_processing(mendeley_df_raw.copy()) 
        
        if mendeley_avg_composition is None:
            print("No se pudo calcular la composición promedio. Saliendo del programa.")
        else:
            print("\nComposición Promedio por Tipo de Procesamiento (solo Al y Total_Impurities en %):")
            print(mendeley_avg_composition[['Al', 'Total_Impurities']].round(3)) 

            # 3. Generar datos sintéticos para entrenamiento del ML
            X_synthetic, Y_synthetic = generate_synthetic_data(
                mendeley_avg_composition, 
                n_samples=N_SAMPLES_SYNTHETIC, 
                fixed_total_kg=TRAINING_UNIT_KG,
                noise_level=NOISE_LEVEL
            )

            if X_synthetic is None:
                print("No se pudieron generar datos sintéticos. Saliendo.")
            else:
                # 4. Dividir los datos sintéticos en conjuntos de entrenamiento y prueba
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X_synthetic, Y_synthetic, test_size=0.2, random_state=42
                )
                print(f"\nDatos divididos: Entrenamiento {X_train.shape[0]} muestras, Prueba {X_test.shape[0]} muestras.")

                # 5. Entrenar los modelos de Regresión Lineal
                trained_models = train_ml_models(X_train, Y_train)

                # 6. Evaluar los modelos (opcional, pero buena práctica)
                evaluate_models(trained_models, X_test, Y_test)
                
                print("\n--- Predicción de Output de Fundición con Modelo ML ---")

                # --- PARÁMETROS DE ENTRADA DEL USUARIO ---
                total_material_input_kg = float(input("Ingrese la cantidad TOTAL de material a fundir en KG: "))
                
                print("\nIngrese los porcentajes de cada tipo de material (la suma debe ser 100%):")
                pct_estructural_struts = float(input("  Porcentaje de 'Structural Struts': "))
                pct_overwrap = float(input("  Porcentaje de 'Overwrap': "))
                pct_drink_pouch = float(input("  Porcentaje de 'Drink Pouch': "))
                pct_rehydratable_pouch = float(input("  Porcentaje de 'Rehydratable Pouch': "))
                
                # 7. Realizar la predicción final utilizando los modelos ML entrenados
                kilos_desechos_impurezas, kilos_aluminio_util, kilos_otros_residuos = predict_al_output_with_ml(
                    cantidad_total_material_kg=total_material_input_kg,
                    porcentaje_estructural_struts=pct_estructural_struts,
                    porcentaje_overwrap=pct_overwrap,
                    porcentaje_drink_pouch=pct_drink_pouch,
                    porcentaje_rehydratable_pouch=pct_rehydratable_pouch,
                    trained_models=trained_models,
                    training_unit_kg=TRAINING_UNIT_KG # Pasar la unidad de entrenamiento
                )
                print(f"\n--- Resumen de Predicción ML ---")
                print(f"Para {total_material_input_kg:.2f} kg de material ingresado:")
                print(f"  Se esperan {kilos_aluminio_util:.3f} kg de aluminio útil fundido.")
                print(f"  Se esperan {kilos_desechos_impurezas:.3f} kg de desechos (impurezas/otros aleantes fundidos).")
                print(f"  Se esperan {kilos_otros_residuos:.3f} kg de otros residuos/pérdidas.")
