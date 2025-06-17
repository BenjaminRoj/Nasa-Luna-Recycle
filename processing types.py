import pandas as pd

def get_processing_types_from_file(file_path):
    """
    Lee un archivo de texto (que se asemeja a un CSV) y extrae los diferentes
    tipos de 'Processing' de la primera columna.

    Args:
        file_path (str): La ruta al archivo CSV.

    Returns:
        set: Un conjunto de cadenas que representan los tipos únicos de 'Processing'.
             Retorna None si hay un error al leer el archivo.
    """
    processing_types = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Saltar la primera línea (cabecera)
            header = f.readline()
            
            for line in f:
                line = line.strip() # Eliminar espacios en blanco y saltos de línea
                if not line: # Saltar líneas vacías
                    continue

                # Intentar encontrar la primera coma que separa el número de fila del texto de 'Processing'
                # Y luego la segunda coma que separa 'Processing' de los datos numéricos.
                # Si el formato es inconsistente, esto podría necesitar ajuste.
                
                parts = line.split(',', 2) # Divide la línea en máximo 3 partes: número_fila, processing_text, resto_datos
                
                if len(parts) >= 2:
                    # La segunda parte (parts[1]) debería ser el texto de 'Processing'
                    # Ejemplo: 'Solutionised + Artificially peak aged'
                    processing_text = parts[1].strip()
                    processing_types.add(processing_text)
                else:
                    print(f"Advertencia: La línea no tiene el formato esperado y será omitida: {line[:50]}...") # Imprimir un fragmento de la línea problemática

        return processing_types

    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta '{file_path}'")
        return None
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        return None

# --- Cómo usar el código ---

if __name__ == "__main__":
    # ¡IMPORTANTE! Reemplaza 'tu_archivo.csv' con la ruta real a tu archivo.
    # Por ejemplo: 'datos_aluminio.csv' o 'C:/Users/TuUsuario/Documentos/datos.csv'
    file_path = 'C:/Users/benja/Downloads/al_data.csv' # Asume que el archivo se llama Aluminium_Data.csv y está en la misma carpeta

    possible_processing_fields = get_processing_types_from_file(file_path)

    if possible_processing_fields:
        print("Campos posibles en 'Processing':")
        for field in sorted(list(possible_processing_fields)):
            print(f"- {field}")
        print(f"\nNúmero total de tipos de 'Processing' encontrados: {len(possible_processing_fields)}")
    else:
        print("No se pudieron extraer los campos de 'Processing'. Revisa la ruta del archivo y su formato.")
