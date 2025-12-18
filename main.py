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
import numpy as np
import matplotlib.pyplot as plt

Historys = []
Barras = []
BS = []

def redondear_abajo_unidad(x):
    return math.floor(x)

def redondear_abajo_decima(x):
    return math.floor(x * 10) / 10

def redondear_arriba_decima(x):
    return math.ceil(x * 10) / 10

def plot_Barras(datos, labels, nombre_parametro, titulo="Resultados del entrenamiento"):

    accuracies = [d[0] for d in datos]
    tiempos    = [d[1] for d in datos]

    n = len(datos)
    x = np.arange(n)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    tiempo_min = min(tiempos)
    tiempo_max = max(tiempos)

    ymin_tiempo = redondear_abajo_unidad(tiempo_min - 10)
    ymax_tiempo = tiempo_max * 1.05

    ax1.bar(
        x - width/2,
        tiempos,
        width,
        label="Tiempo (s)",
        alpha=0.7,
        color="tab:orange"
    )
    ax1.set_ylabel("Tiempo (s)")
    ax1.set_ylim(ymin_tiempo, ymax_tiempo)

    acc_min = min(accuracies)
    acc_max = max(accuracies)

    ymin_acc = redondear_abajo_decima(acc_min)
    ymax_acc = redondear_arriba_decima(acc_max) + 0.1

    ax2 = ax1.twinx()
    ax2.bar(
        x + width/2,
        accuracies,
        width,
        label="Accuracy",
        alpha=0.7,
        color="purple"
    )
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(ymin_acc, ymax_acc)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(nombre_parametro)

    plt.title(titulo)

    plt.tight_layout()
    plt.show()

 
def plot_curva(Y, titulo, xscale = "linear", yscale = "linear"):
    plt.title(titulo)
    plt.plot(Y)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show()     


def plot_Historys(Historys):

    train_acc = np.array([h.history['accuracy'] for h in Historys])
    val_acc   = np.array([h.history['val_accuracy'] for h in Historys])
    train_loss = np.array([h.history['loss'] for h in Historys])
    val_loss   = np.array([h.history['val_loss'] for h in Historys])

    mean_train_acc = train_acc.mean(axis=0)
    mean_val_acc   = val_acc.mean(axis=0)
    mean_train_loss = train_loss.mean(axis=0)
    mean_val_loss   = val_loss.mean(axis=0)

    epochs = range(1, len(mean_train_acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, mean_train_acc, label='train_accuracy')
    plt.plot(epochs, mean_val_acc, '--', label='val_accuracy')
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Accuracy media por época")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mean_train_loss, label='train_loss')
    plt.plot(epochs, mean_val_loss, '--', label='val_loss')
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss media por época")
    plt.legend()

    plt.tight_layout()
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

def tarea_1(X_train, X_test, Y_train, Y_test):
    model = keras.Sequential(
        [
            layers.Dense(48, activation="sigmoid", name="layer1"),
            layers.Dense(10, activation="softmax", name="layer2"),
        ]
    )
    
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    model.summary()
    
    history = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.1)
    
    test_scores = model.evaluate(X_test, Y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    
def tarea_2(X_train, X_test, Y_train, Y_test):

    N_RUNS = 5        # puedes subirlo a 3–5 luego
    EPOCHS = 100

    for i in range(N_RUNS):

        print(f"Entrenamiento número {i+1}")
        keras.backend.clear_session()

        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(48, activation="sigmoid"),
            layers.Dense(10, activation="softmax"),
        ])

        model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train, Y_train,
            batch_size=32,
            epochs=EPOCHS,
            validation_split=0.1,
            verbose=1
        )

        Historys.append(history)

        test_scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

    plot_Historys(Historys)
    
def tarea_3(X_train, X_test, Y_train, Y_test):
    for i in range(5):

        model = keras.Sequential(
            [
                keras.Input(X_train[0].shape),
                layers.Dense(48, activation="sigmoid", name="layer1"),
                layers.Dense(10, activation="softmax", name="layer2"),
            ]
        )

        model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()

        my_callbacks = [
            keras.callbacks.EarlyStopping(
                patience=70,
                monitor="accuracy",
                min_delta=0.1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='model.{epoch:02d}-{val_loss:.2f}.keras'
            ),
            keras.callbacks.TensorBoard(log_dir='./logs'),
        ]

        print(f"Entrenamiento numero {i+1}")
        bs = 2**(i+6)
        BS.append(bs)
        print("batch_size = ", bs)

        ini = time.time()
        Historys.append(
            model.fit(
                X_train,
                Y_train,
                batch_size=bs,
                epochs=25,
                validation_split=0.1,
                callbacks=my_callbacks
            )
        )
        fin = time.time()

        t = fin - ini

        test_scores = model.evaluate(X_test, Y_test, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

        Barras.append((test_scores[1], t))

    plot_Barras(
        Barras,
        BS,
        nombre_parametro="batch_size",
        titulo="accuracy y tiempo vs batch size"
    )

    


def main():
    
    X_train, X_test, Y_train, Y_test = cargar_y_preprocesar_cifar10()
    #tarea_1(X_train, X_test, Y_train, Y_test)
    #tarea_2(X_train, X_test, Y_train, Y_test)
    tarea_3(X_train, X_test, Y_train, Y_test)


if __name__ == "__main__":
    main()
