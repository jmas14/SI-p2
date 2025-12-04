'''
Eliminar los mensajes de error por no instalar el soporte de CUDA
'''
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from keras import layers
from keras import ops
from keras import activations
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import time

def show_image(imagen, titulo):
    plt.figure()
    plt.suptitle(titulo)
    plt.imshow(imagen, cmap = "Greys")
    plt.show()
 
def plot_curva(Y, titulo, xscale = "linear", yscale = "linear"):
    plt.title(titulo)
    plt.plot(Y)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show()     

def plot_Historys(Historys):
    plt.figure(figsize=(12, 5))

    # ==== ACCURACY ====
    plt.subplot(1, 2, 1)
    for h in Historys:
        acc = h.history['accuracy']
        val_acc = h.history['val_accuracy']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc,  '-',  alpha=0.4)   # train_accuracy
        plt.plot(epochs, val_acc, '--', alpha=0.4) # val_accuracy

    plt.title("Accuracy por época")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend(["train_accuracy", "val_accuracy"])

    # ==== LOSS ====
    plt.subplot(1, 2, 2)
    for h in Historys:
        loss = h.history['loss']
        val_loss = h.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss,  '-',  alpha=0.4)   # train_loss
        plt.plot(epochs, val_loss, '--', alpha=0.4) # val_loss

    plt.title("Loss por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "val_loss"])

    plt.tight_layout()
    plt.show()

def plot_Barras(datos, labels, nombre_parametro, titulo="Resultados del experimento"):
    """
    datos: lista de tuplas (accuracy, time)
    labels: lista de textos que describen el valor del parámetro en cada experimento
    nombre_parametro: etiqueta para el eje X
    titulo: título de la gráfica
    """

    accuracies = [d[0] for d in datos]
    tiempos = [d[1] for d in datos]

    n = len(datos)
    x = np.arange(n)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # === EJE IZQUIERDO → Tiempo (NARANJA) ===
    barras_tiempo = ax1.bar(
        x - width/2,
        tiempos,
        width,
        label="Tiempo (s)",
        alpha=0.7,
        color="tab:orange"
    )
    ax1.set_ylabel("Tiempo (s)")
    ax1.set_ylim(0, max(tiempos) * 1.2)

    # === EJE DERECHO → Accuracy (MORADO) ===
    ax2 = ax1.twinx()
    barras_acc = ax2.bar(
        x + width/2,
        accuracies,
        width,
        label="Accuracy",
        alpha=0.7,
        color="purple"
    )
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)

    # === Etiquetas eje X ===
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(nombre_parametro)

    # === Título ===
    plt.title(titulo)

    # === Etiquetas numéricas ===
    for rect, t in zip(barras_tiempo, tiempos):
        ax1.text(rect.get_x() + rect.get_width()/2,
                 t + max(tiempos)*0.02,
                 f"{t:.2f}",
                 ha='center', va='bottom', fontsize=9)

    for rect, acc in zip(barras_acc, accuracies):
        ax2.text(rect.get_x() + rect.get_width()/2,
                 acc + 0.02,
                 f"{acc:.3f}",
                 ha='center', va='bottom', fontsize=9)

    # === Leyendas combinadas ===
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.show()
    
    
def cargar_y_preprocesar_cifar10():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    # Normalización a [0, 1]
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Aplanado: de (32, 32, 3) → (3072)
    X_train = X_train.reshape((X_train.shape[0], 32 * 32 * 3))
    X_test = X_test.reshape((X_test.shape[0], 32 * 32 * 3))
    
    # convertir cada categoria en arrays de 10 posiciones donde el 1 indica el valor  
    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)

    return X_train, X_test, Y_train, Y_test

def main():

    X_train, X_test, Y_train, Y_test = cargar_y_preprocesar_cifar10()

    Historys = []
    Barras = []
    num_neuronas = ["400x300x200x50x30x20",
                    "150x200x300x200x100x50",
                    "50x60x90x200x300x300",
                    "100x200x300x200x150x50",
                    "300x150x50x50x150x300"
                    ]
    for i in range(5):
        if i == 0:
            model = keras.Sequential(
                [
                    keras.Input(X_train[0].shape),
                    layers.Dense(400, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer1"),
                    layers.Dense(300, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer2"),
                    layers.Dense(200, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer3"),
                    layers.Dense(50,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer4"),
                    layers.Dense(30,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer5"),
                    layers.Dense(20,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer6"),
                    layers.Dense(10,  activation="softmax", name="layer7"),
                ]
            )

        if i == 1:
            model = keras.Sequential(
                [
                    keras.Input(X_train[0].shape),
                    layers.Dense(150, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer1"),
                    layers.Dense(200, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer2"),
                    layers.Dense(300, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer3"),
                    layers.Dense(200, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer4"),
                    layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer5"),
                    layers.Dense(50,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer6"),
                    layers.Dense(10, activation="softmax", name="layer7"),
                ]
            )

        if i == 2:
            model = keras.Sequential(
                [
                    keras.Input(X_train[0].shape),
                    layers.Dense(50,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer1"),
                    layers.Dense(60,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer2"),
                    layers.Dense(90,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer3"),
                    layers.Dense(200, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer4"),
                    layers.Dense(300, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer5"),
                    layers.Dense(300, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer6"),
                    layers.Dense(10, activation="softmax", name="layer7"),
                ]
            )

        if i == 3:
            model = keras.Sequential(
                [
                    keras.Input(X_train[0].shape),
                    layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer1"),
                    layers.Dense(200, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer2"),
                    layers.Dense(300, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer3"),
                    layers.Dense(200, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer4"),
                    layers.Dense(150, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer5"),
                    layers.Dense(50,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer6"),
                    layers.Dense(10, activation="softmax", name="layer7"),
                ]
            )

        if i == 4:
            model = keras.Sequential(
                [
                    keras.Input(X_train[0].shape),
                    layers.Dense(300, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer1"),
                    layers.Dense(150, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer2"),
                    layers.Dense(50,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer3"),
                    layers.Dense(50,  activation="sigmoid", kernel_initializer="glorot_uniform", name="layer4"),
                    layers.Dense(150, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer5"),
                    layers.Dense(300, activation="sigmoid", kernel_initializer="glorot_uniform", name="layer6"),
                    layers.Dense(10, activation="softmax", name="layer7"),
                ]
            )

        model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()

        my_callbacks = [
            keras.callbacks.EarlyStopping(patience=5, monitor="accuracy", min_delta=0.001),
            keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.keras'),
            keras.callbacks.TensorBoard(log_dir='./logs'),
        ]


        print(f"Entrenamiento numero {i+1}")
        ini = time.time()
        Historys.append(model.fit(X_train, Y_train, batch_size=512, epochs=25, validation_split=0.1,callbacks=my_callbacks))
        fin = time.time()
        t = fin - ini
        test_scores = model.evaluate(X_test, Y_test, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])
        Barras.append((test_scores[1],t))

    plot_Barras(
        Barras,
        num_neuronas,
        nombre_parametro="neuronas_capa_1xneuronas_capa_2",
        titulo="acierto y tiempo con dos capas"
    )
    #plot_Historys(Historys)
    
    
if __name__ == "__main__": main()