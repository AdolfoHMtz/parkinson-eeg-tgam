# Datos / Data

Esta carpeta guarda los datos de entrada y los resultados intermedios. En el repositorio
solo se incluye lo pequeño; lo pesado se regenera con los scripts.
*This folder holds input data and intermediate results. Only small files are versioned;
everything else is regenerated with the scripts.*

| Carpeta | En el repo | Contenido |
|---|---|---|
| `sesiones_grabadas/` | ✅ | 3 sesiones TGAM del autor (raw 512 Hz + bandas ~1 Hz), grabadas con consentimiento |
| `openneuro_procesado/` | ✅ | OpenNeuro **ds002778**, canal **Fp1** remuestreado a **512 Hz** (46 grabaciones, 31 personas) |
| `modelos_entrenados/` | ✅ métricas y gráficas · ❌ `.keras` | Una carpeta por corrida. El modelo final se publica como *GitHub Release* |
| `procesados/` | ❌ (~450 MB por variante) | Ventanas + GAF listos para entrenar |
| `sesiones/` | ❌ | Sesiones de demostración, no usadas en el análisis |

## Regenerar / Regenerate

```bash
# 1. (Opcional) Volver a procesar OpenNeuro desde el original BIDS (~420 MB)
python -c "from src.preprocesamiento.dataset_publico import descargar_dataset, procesar_dataset; \
descargar_dataset('datos/openneuro_raw'); \
procesar_dataset('datos/openneuro_raw', 'datos/openneuro_procesado', solo_fp1=True)"

# 2. Ventanas de 2 s, 50 % de solapamiento, GAF 128x128, división por paciente
python -m src.preprocesamiento.pipeline --fuente publico --rama b

# 3. Entrenar la CNN (Rama B)
python -m src.modelos.entrenar_cnn_rama_b datos/procesados/rama_b_gaf_v2.0s_o50_balanceado/datos.npz
```

## Créditos del dataset / Dataset credit

Rockhill, A. P., Jackson, N., George, J., Aron, A., & Swann, N. C. (2020).
*UC San Diego Resting State EEG Data from Patients with Parkinson's Disease.*
OpenNeuro, ds002778. Licencia CC0.

Cada paciente con Parkinson tiene dos sesiones (`on` y `off` medicación). La división
train/val/test se hace **por persona** para que la misma persona nunca aparezca en dos
conjuntos.
*Each Parkinson's patient has two sessions (on/off medication). The train/val/test split is
done **per person** so the same individual never appears in two sets.*
