#!/usr/bin/env python3
"""
train_dnn_agent.py

Carga data/X.npy, data/y.npy, entrena un MLP usando solo CPU,
muestra barra de progreso con tqdm y guarda el modelo en
briscas/agents/models/briscas_dnn.keras (nuevo formato KerasV3).
"""

import os
# ——————————————————————————————
# Deshabilitar logs y warnings de TensorFlow
# ——————————————————————————————
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tqdm.keras import TqdmCallback
from multiprocessing import cpu_count

# Configurar hilos de CPU
cpus = cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(cpus)
tf.config.threading.set_inter_op_parallelism_threads(cpus)

# ——————————————————————————————
# Paths y parámetros
# ——————————————————————————————
DATA_DIR       = "data"
MODEL_DIR      = "briscas/agents/models"
MODEL_PATH     = os.path.join(MODEL_DIR, "briscas_dnn.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

HIDDEN_UNITS     = 64
NUM_CLASSES      = 40
BATCH_SIZE       = 32
EPOCHS           = 100
LR               = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT       = 0.15

# ——————————————————————————————
# Cargar y preparar datos
# ——————————————————————————————
def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def make_dataset(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ——————————————————————————————
# Modelo
# ——————————————————————————————
def build_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(HIDDEN_UNITS, activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(HIDDEN_UNITS, activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    m   = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    return m

# ——————————————————————————————
# Entrenamiento
# ——————————————————————————————
def main():
    X, y     = load_data()
    n        = len(X)
    n_val    = int(n * VALIDATION_SPLIT)
    n_test   = int(n * TEST_SPLIT)
    n_train  = n - n_val - n_test

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_v,  y_v  = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_te, y_te = X[-n_test:], y[-n_test:]

    ds_tr = make_dataset(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    ds_v  = make_dataset(X_v,  y_v,  BATCH_SIZE)
    ds_te = make_dataset(X_te, y_te, BATCH_SIZE)

    model = build_model(X.shape[1])

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor="val_loss",
            save_format="keras"
        ),
        TqdmCallback(verbose=1)
    ]

    # Entrenar forzando CPU
    with tf.device("/CPU:0"):
        model.fit(
            ds_tr,
            validation_data=ds_v,
            epochs=EPOCHS,
            verbose=0,
            callbacks=cbs
        )
        loss, acc, top3 = model.evaluate(ds_te, verbose=0)

    print(f"\n➡ Test loss: {loss:.4f}  acc: {acc:.4f}  top-3: {top3:.4f}")
    print(f"✔ Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    main()
