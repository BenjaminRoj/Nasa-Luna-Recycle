import numpy as np

# --- NUEVAS CONSTANTES PARA PLÁSTICOS ---
# PORCENTAJES DE RENDIMIENTO PARA EL MODELO LINEAL SIMPLIFICADO
# Ahora la asunción es del 50% de material útil.
PERCENTAGE_USEFUL_PLASTIC = 50.0 # % del input total que se convierte en plástico útil reciclado
PERCENTAGE_TOTAL_WASTE = 100.0 - PERCENTAGE_USEFUL_PLASTIC # % del input total que son desechos


def calculate_plastic_output_linearly(cantidad_total_material_kg: float):
    """
    Calcula la cantidad de plástico útil y desechos totales utilizando una relación lineal simple.
    """
    print("\n--- Resultados del Modelo Lineal Simple para Plásticos ---")
    
    kilos_plastico_util = cantidad_total_material_kg * (PERCENTAGE_USEFUL_PLASTIC / 100.0)
    kilos_desechos_totales = cantidad_total_material_kg * (PERCENTAGE_TOTAL_WASTE / 100.0)
    
    print(f"Cantidad Total de Material Plástico Ingresado: {cantidad_total_material_kg:.2f} kg")
    print(f"Cantidad Estimada de Plástico Útil Reciclable: {kilos_plastico_util:.3f} kg ({PERCENTAGE_USEFUL_PLASTIC:.1f}%)")
    print(f"Cantidad Estimada de Desechos Totales: {kilos_desechos_totales:.3f} kg ({PERCENTAGE_TOTAL_WASTE:.1f}%)")
    
    total_output_sum_check = kilos_plastico_util + kilos_desechos_totales
    print(f"Suma Total de Outputs: {total_output_sum_check:.3f} kg. (Debería ser ~igual al total ingresado)")
    
    return kilos_plastico_util, kilos_desechos_totales


# --- PROCESO PRINCIPAL ---

if __name__ == "__main__":
    print("\n--- Modelado Simplificado para Reciclaje de Plásticos a Filamento ---")
    print("Este modelo utiliza una relación lineal simple basada en porcentajes de rendimiento fijos.")
    print(f"Asunción de rendimiento: {PERCENTAGE_USEFUL_PLASTIC:.1f}% Plástico Útil, {PERCENTAGE_TOTAL_WASTE:.1f}% Desechos Totales.")

    # --- PARÁMETROS DE ENTRADA DEL USUARIO ---
    total_material_input_kg = float(input("Ingrese la cantidad TOTAL de material plástico de desecho a procesar en KG: "))
    
    # Calcular la cantidad de plástico útil y desechos usando la relación lineal
    kilos_plastico_util, kilos_desechos_totales = calculate_plastic_output_linearly(
        cantidad_total_material_kg=total_material_input_kg
    )

    print(f"\n--- Resumen Final del Proceso de Reciclaje de Plástico ---")
    print(f"Para {total_material_input_kg:.2f} kg de material plástico de desecho ingresado:")
    print(f"  Kilos de Plástico Útil Reciclado (apto para filamento): {kilos_plastico_util:.3f} kg")
    print(f"  Kilos de Desechos Totales Generados (no aptos para filamento): {kilos_desechos_totales:.3f} kg")
