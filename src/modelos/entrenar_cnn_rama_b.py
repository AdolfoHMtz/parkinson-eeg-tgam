import os
import argparse
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

# Descomentar/instalar tensorflow en requirements.txt para usar este script
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def cargar_datos_rama_b(ruta_npz: str):
    """Carga y prepara los datos de la Rama B para Keras."""
    datos = np.load(ruta_npz, allow_pickle=True)
    
    # Las imagenes GAF vienen como (N, resolucion, resolucion)
    # Keras espera canales al final: (N, resolucion, resolucion, 1)
    X_train = np.expand_dims(datos["X_train"], axis=-1)
    X_val = np.expand_dims(datos["X_val"], axis=-1)
    X_test = np.expand_dims(datos["X_test"], axis=-1)
    
    y_train = datos["y_train"]
    y_val = datos["y_val"]
    y_test = datos["y_test"]
    
    print(f"Shapes cargados:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def imprimir_distribucion_clases(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
    """Muestra el balance de clases para contextualizar las metricas."""
    print("Distribucion de clases:")
    for nombre, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        conteo = dict(zip(*np.unique(y, return_counts=True)))
        total = len(y)
        clase_0 = int(conteo.get(0, 0))
        clase_1 = int(conteo.get(1, 0))
        print(
            f"  {nombre}: clase 0={clase_0} ({clase_0/total:.4f}), "
            f"clase 1={clase_1} ({clase_1/total:.4f})"
        )

def crear_modelo_cnn(input_shape):
    """
    Crea una CNN compacta con regularizacion para mejorar la generalizacion.
    """
    modelo = models.Sequential([
        layers.Input(shape=input_shape),
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Reducimos parametros para evitar que memorice ruido espacial
        layers.GlobalAveragePooling2D(),
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
        layers.Dropout(0.35),
        layers.Dense(1, activation='sigmoid')  # 1 salida: 0 (sano) a 1 (parkinson)
    ])
    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='roc_auc'),
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return modelo

def graficar_historial(historia, ruta_salida):
    """Guarda las curvas de aprendizaje."""
    acc = historia.history['accuracy']
    val_acc = historia.history['val_accuracy']
    loss = historia.history['loss']
    val_loss = historia.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Grafica de exactitud
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Entrenamiento')
    plt.plot(epochs, val_acc, 'r-', label='Validacion')
    plt.title('Exactitud')
    plt.legend()
    
    # Grafica de perdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Entrenamiento')
    plt.plot(epochs, val_loss, 'r-', label='Validacion')
    plt.title('Perdida (Loss)')
    plt.legend()
    
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    print(f"  Historial guardado en {ruta_salida}")


def graficar_matriz_confusion(matriz: list[list[int]], ruta_salida: str):
    """Guarda una matriz de confusion como imagen."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title("Matriz de confusion")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sano", "Parkinson"])
    ax.set_yticklabels(["Sano", "Parkinson"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i][j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matriz de confusion guardada en {ruta_salida}")


def guardar_resumen_metricas(ruta_salida: str, payload: dict):
    """Guarda un resumen serializable de la corrida."""
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Resumen de metricas guardado en {ruta_salida}")


def buscar_mejor_umbral(y_true: np.ndarray, y_prob: np.ndarray):
    """Busca el umbral que maximiza balanced accuracy, usando F1 como desempate."""
    mejor = {
        "threshold": 0.5,
        "f1": -1.0,
        "balanced_accuracy": -1.0,
    }

    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= threshold).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        if (bal_acc > mejor["balanced_accuracy"]) or (bal_acc == mejor["balanced_accuracy"] and f1 > mejor["f1"]):
            mejor = {
                "threshold": float(threshold),
                "f1": float(f1),
                "balanced_accuracy": float(bal_acc),
            }

    return mejor


def evaluar_con_umbral(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    """Calcula metricas utiles con un umbral dado."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def main():
    parser = argparse.ArgumentParser(description="Entrenar CNN con datos de Rama B (GAF)")
    parser.add_argument("ruta_npz", help="Ruta al archivo datos.npz de Rama B")
    parser.add_argument("--epochs", type=int, default=50, help="Numero de epocas max")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamano de batch")
    parser.add_argument("--salida", default="datos/modelos_entrenados/cnn_rama_b", help="Carpeta de salida")
    args = parser.parse_args()
    
    # Crear versionado automatico con fecha y hora para NO sobreescribir modelos ni graficas
    marca_tiempo = datetime.datetime.now().strftime("v%Y%m%d_%H%M%S")
    carpeta_salida_versionada = f"{args.salida}_{marca_tiempo}"
    os.makedirs(carpeta_salida_versionada, exist_ok=True)
    
    # 1. Cargar datos
    print("1. Cargando datos...")
    X_train, y_train, X_val, y_val, X_test, y_test = cargar_datos_rama_b(args.ruta_npz)
    
    # 2. Crear modelo
    print("\n2. Construyendo modelo CNN...")
    modelo = crear_modelo_cnn(input_shape=X_train.shape[1:])
    modelo.summary()
    imprimir_distribucion_clases(y_train, y_val, y_test)
    
    # 3. Callbacks para Early Stopping, ajuste dinamico del LR y guardado del mejor modelo
    ruta_mejor_modelo = os.path.join(carpeta_salida_versionada, "mejor_modelo.keras")
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(ruta_mejor_modelo, monitor='val_pr_auc', mode='max', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_pr_auc',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    # 4. Entrenar
    print("\n3. Entrenando modelo...")
    historia = modelo.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 5. Calibrar umbral en validacion y evaluar en test
    print("\n4. Calibrando umbral con validacion...")
    val_prob = modelo.predict(X_val, verbose=0).ravel()
    mejor_umbral = buscar_mejor_umbral(y_val, val_prob)
    print(
        f"  Umbral optimo: {mejor_umbral['threshold']:.2f} | "
        f"F1 val: {mejor_umbral['f1']:.4f} | "
        f"Balanced Acc val: {mejor_umbral['balanced_accuracy']:.4f}"
    )

    print("\n5. Evaluacion final en conjunto de Test...")
    test_prob = modelo.predict(X_test, verbose=0).ravel()
    resultados_test = modelo.evaluate(X_test, y_test, verbose=0, return_dict=True)
    for nombre_metrica, valor in resultados_test.items():
        if isinstance(valor, (float, np.floating)):
            print(f"  {nombre_metrica}: {valor:.4f}")
        else:
            print(f"  {nombre_metrica}: {valor}")

    metricas_umbral = evaluar_con_umbral(y_test, test_prob, mejor_umbral["threshold"])
    print("  --- Metricas con umbral calibrado ---")
    print(f"  threshold: {metricas_umbral['threshold']:.2f}")
    print(f"  accuracy: {metricas_umbral['accuracy']:.4f}")
    print(f"  balanced_accuracy: {metricas_umbral['balanced_accuracy']:.4f}")
    print(f"  precision: {metricas_umbral['precision']:.4f}")
    print(f"  recall: {metricas_umbral['recall']:.4f}")
    print(f"  f1: {metricas_umbral['f1']:.4f}")
    print(f"  roc_auc: {metricas_umbral['roc_auc']:.4f}")
    print(f"  pr_auc: {metricas_umbral['pr_auc']:.4f}")
    print(f"  confusion_matrix: {metricas_umbral['confusion_matrix']}")
    
    # 6. Graficar y guardar resultados final
    ruta_grafica = os.path.join(carpeta_salida_versionada, "curvas_aprendizaje.png")
    graficar_historial(historia, ruta_grafica)
    ruta_cm = os.path.join(carpeta_salida_versionada, "matriz_confusion.png")
    graficar_matriz_confusion(metricas_umbral["confusion_matrix"], ruta_cm)

    resumen = {
        "entrada": args.ruta_npz,
        "salida": carpeta_salida_versionada,
        "umbral_validacion": mejor_umbral,
        "metricas_test_raw": {k: v for k, v in resultados_test.items()},
        "metricas_test_umbral_calibrado": metricas_umbral,
        "clases": {
            "train": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "val": {int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            "test": {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        },
    }
    ruta_json = os.path.join(carpeta_salida_versionada, "resumen_metricas.json")
    guardar_resumen_metricas(ruta_json, resumen)
    
    # Guardar modelo final
    modelo.save(os.path.join(carpeta_salida_versionada, "modelo_final.keras"))
    print(f"\nEntrenamiento finalizado. Modelo y graficas guardados en: {carpeta_salida_versionada}")

if __name__ == "__main__":
    main()
