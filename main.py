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
import math
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

Historys = []
Barras = []

def truncar_unidad(x):
    return int(x)

def redondear_abajo_decima(x):
    return math.floor(x * 10) / 10

def redondear_arriba_decima(x):
    return math.ceil(x * 10) / 10

def plot_Barras(datos, labels, f_parametro, titulo="Resultados del entrenamiento"):

    accuracies = [d[0] for d in datos]
    tiempos    = [d[1] for d in datos]

    n = len(datos)
    x = np.arange(n)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    tiempo_min = min(tiempos)
    tiempo_max = max(tiempos)

    ymin_tiempo = truncar_unidad(tiempo_min) - 5
    ymax_tiempo = truncar_unidad(tiempo_max) + 5

    barras_tiempo = ax1.bar(
        x - width/2,
        tiempos,
        width,
        label="Tiempo (s)",
        alpha=0.7,
        color="tab:orange"
    )
    ax1.set_ylabel("Tiempo (s)")
    ax1.set_ylim(ymin_tiempo, ymax_tiempo)

    for barra in barras_tiempo:
        altura = barra.get_height()
        ax1.text(
            barra.get_x() + barra.get_width() / 2,
            altura,
            f"{altura:.1f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    acc_min = min(accuracies)
    acc_max = max(accuracies) 

    ymin_acc = redondear_abajo_decima(acc_min)
    ymax_acc = redondear_arriba_decima(acc_max) 

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
    ax2.set_ylim(ymin_acc, ymax_acc)

    for barra in barras_acc:
        altura = barra.get_height()
        ax2.text(
            barra.get_x() + barra.get_width() / 2,
            altura,
            f"{altura:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(f_parametro)

    plt.title(titulo)
    plt.tight_layout()
    plt.show()
 
def plot_curva(Y, titulo, xscale = "linear", yscale = "linear"):
    plt.title(titulo)
    plt.plot(Y)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show()     

'''
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
'''

def plot_Historys(Historys):

    min_epochs = min(len(h.history['accuracy']) for h in Historys)

    train_acc = np.array([h.history['accuracy'][:min_epochs] for h in Historys])
    val_acc   = np.array([h.history['val_accuracy'][:min_epochs] for h in Historys])
    train_loss = np.array([h.history['loss'][:min_epochs] for h in Historys])
    val_loss   = np.array([h.history['val_loss'][:min_epochs] for h in Historys])

    mean_train_acc = train_acc.mean(axis=0)
    mean_val_acc   = val_acc.mean(axis=0)
    mean_train_loss = train_loss.mean(axis=0)
    mean_val_loss   = val_loss.mean(axis=0)

    epochs = range(1, min_epochs + 1)

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

def cargar_y_preprocesar_cifar10_cnn():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test  = to_categorical(Y_test, num_classes=10)

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
    
def tarea_2(X_train, X_test, Y_train, Y_test, n_runs=5, epochs=100):

    for i in range(n_runs):

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

        model.summary()

        history = model.fit(
            X_train,
            Y_train,
            batch_size=32,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )

        Historys.append(history)

        test_scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

    plot_Historys(Historys)
    
def tarea_3(X_train, X_test, Y_train, Y_test):
    
    BS = []
    
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
        f_parametro="batch_size",
        titulo="accuracy y tiempo vs batch size"
    )

def tarea_4(X_train, X_test, Y_train, Y_test, n_entrenamientos=5):
    
    funciones ={
        "relu": {"activacion": "relu","ini": "he_normal"},
        "relu6": {"activacion": "relu6","ini": "he_normal"},
        "sigmoid": {"activacion": "sigmoid","ini": "glorot_uniform"},
        "hard_sigmoid": {"activacion": keras.activations.hard_sigmoid,"ini": "glorot_uniform"},
        "swish": {"activacion": "swish","ini": "glorot_uniform"}
    }

    resultados = []  # (f_función, acc_media, tiempo_medio)

    for f, f_keras in funciones.items():

        aciertos = []
        tiempos = []

        print(f"Función de activación: {f}")

        for i in range(n_entrenamientos):
            print(f"entrenamiento {i + 1}/{n_entrenamientos}")

            model = keras.Sequential([
                keras.Input(shape=X_train[0].shape),
                layers.Dense(
                    48,
                    activation=f_keras["activacion"],
                    kernel_initializer=f_keras["ini"]
                ),
                layers.Dense(10, activation="softmax")
            ])

            model.compile(
                optimizer="Adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            my_callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=70,
                    monitor="accuracy",
                    min_delta=0.1
                )
            ]

            ini = time.time()
            model.fit(
                X_train,
                Y_train,
                batch_size=512,
                epochs=25,
                validation_split=0.1,
                callbacks=my_callbacks,
                verbose=0
            )
            fin = time.time()

            t = fin - ini
            test_scores = model.evaluate(X_test, Y_test, verbose=0)

            aciertos.append(test_scores[1])
            tiempos.append(t)

        a_media = np.mean(aciertos)
        t_medio = np.mean(tiempos)

        resultados.append((a_media, t_medio))

        print(f"Acierto medio: {a_media:.4f}")
        print(f"Tiempo medio: {t_medio:.2f}s")

    plot_Barras(
        resultados,
        funciones.keys(),
        f_parametro="funciones",
        titulo="accuracy y tiempo con diferentes funciones de activacion"
    )

def tarea_5(X_train, X_test, Y_train, Y_test, n_runs=5):
    
    Neuronas = []
    
    for i in range(5):
        
        print(f"Entrenamiento numero {i+1}")
        n = 2**(i+6)
        Neuronas.append(n)
        print("neuronas = ", n)

        model = keras.Sequential(
            [
                keras.Input(X_train[0].shape),
                layers.Dense(n, activation="sigmoid", name="layer1"),
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

        ini = time.time()
        Historys.append(
            model.fit(
                X_train,
                Y_train,
                batch_size=512,
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
        Neuronas,
        f_parametro="neuronas",
        titulo="accuracy y tiempo vs cantidad de neuronas en una capa"
    )
    
def tarea_6(X_train, X_test, Y_train, Y_test, arquitecturas=None,n_repeticiones = 1):
    
    resultados = []
    
    if arquitecturas is None:
        arquitecturas = ["251x166x83x83x166x251","80x120x300x300x120x80","80x300x120x120x300x80","100x100x150x150x250x250","167x167x167x167x166x166"]

    for i, capas in enumerate(arquitecturas):

        print(f"Arquitectura {i + 1}/{len(arquitecturas)}: {capas}")

        aciertos = []
        tiempos = []

        capas = [int(n) for n in capas.split("x")]

        for j in range(n_repeticiones):
            print(f"  Entrenamiento {j + 1}/{n_repeticiones}")

            model = keras.Sequential()
            model.add(keras.Input(shape=X_train[0].shape))

            for n in capas:
                model.add(
                    layers.Dense(
                        n,
                        activation="sigmoid",
                        kernel_initializer="glorot_uniform"
                    )
                )

            model.add(layers.Dense(10, activation="softmax"))

            model.compile(
                optimizer="Adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            my_callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=70,
                    monitor="accuracy",
                    min_delta=0.1
                )
            ]

            ini = time.time()
            model.fit(
                X_train,
                Y_train,
                batch_size=512,
                epochs=25,
                validation_split=0.1,
                callbacks=my_callbacks,
                verbose=0
            )
            fin = time.time()

            test_scores = model.evaluate(X_test, Y_test, verbose=0)

            aciertos.append(test_scores[1])
            tiempos.append(fin - ini)

        a_media = np.mean(aciertos)
        t_medio = np.mean(tiempos)

        resultados.append((a_media, t_medio))

        print(f"Acierto medio: {a_media:.4f}")
        print(f"Tiempo medio: {t_medio:.2f}s")

    plot_Barras(
        resultados,
        arquitecturas,
        f_parametro="neuronas por capa",
        titulo="accuracy y tiempo con diferentes 6 capas y 1000 neuronas"
    )

def tarea_7(X_train, X_test, Y_train, Y_test):

    arquitecturas = {
        "300x700": [300, 700],
        "90x90x182x182x273": [90, 90, 182, 182, 273],
        "90x182x273x182x90": [90, 182, 273, 182, 90],
        "400x600": [400, 600],
        "200x200x200x200x200": [200, 200, 200, 200, 200]
    }

    activation = "sigmoid"
    batch_size = 512
    epochs = 25

    resultados = []

    for nombre, capas in arquitecturas.items():

        print(f"Entrenando arquitectura: {nombre}")
        print(f"Capas: {capas}")

        model = keras.Sequential()
        model.add(keras.Input(shape=X_train[0].shape))

        for n in capas:
            model.add(layers.Dense(
                n,
                activation=activation,
                kernel_initializer="glorot_uniform",
                kernel_regularizer=regularizers.l2(1e-6)
            ))
            model.add(layers.Dropout(0.1))

        model.add(layers.Dense(10, activation="softmax"))

        model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        my_callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=70,
                    monitor="accuracy",
                    min_delta=0.1
                )
            ]
        
        ini = time.time()
        history = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=my_callbacks,
            verbose=0
        )
        fin = time.time()
        t = fin - ini

        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        epocas_reales = len(history.history["loss"])

        resultados.append((test_acc, t))

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Épocas entrenadas: {epocas_reales}")

    plot_Barras(
        resultados,
        arquitecturas.values(),
        f_parametro="arquitecturas",
        titulo="accuracy y tiempo con diferentes capas con 1000 neuronas"
    )

def cnn1(X_train, X_test, Y_train, Y_test, n_runs=5, epochs=100):

    for i in range(n_runs):

        print(f"\nEntrenamiento número {i+1}")
        keras.backend.clear_session()

        model = keras.Sequential([
            
            layers.Input(shape=(32, 32, 3)),
            
            layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_normal"
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_normal"
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(100, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, Y_train,
            batch_size=32,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )

        Historys.append(history)

        test_scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

    plot_Historys(Historys)
    
def cnn2(X_train, X_test, Y_train, Y_test, n_runs=1, epochs=25):

    kernels = kernels = [(1,1), (3,3), (5,5), (3,5), (5,3)]

    resultados = []
    etiquetas = []

    for k in kernels:

        print(f"\nKernel Conv2D: {k[0]}x{k[1]}")
        etiquetas.append(f"{k[0]}x{k[1]}")

        aciertos = []
        tiempos = []

        for i in range(n_runs):
            print(f"  Entrenamiento {i+1}/{n_runs}")
            keras.backend.clear_session()

            model = keras.Sequential([
                layers.Input(shape=(32, 32, 3)),

                layers.Conv2D(
                    filters=16,
                    kernel_size=k,
                    activation="relu",
                    kernel_initializer="he_normal"
                ),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Conv2D(
                    filters=32,
                    kernel_size=k,
                    activation="relu",
                    kernel_initializer="he_normal"
                ),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Flatten(),
                layers.Dense(100, activation="relu"),
                layers.Dense(10, activation="softmax")
            ])

            model.compile(
                optimizer="Adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            ini = time.time()
            model.fit(
                X_train,
                Y_train,
                batch_size=32,
                epochs=epochs,
                validation_split=0.1,
                verbose=0
            )
            fin = time.time()

            test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

            aciertos.append(test_acc)
            tiempos.append(fin - ini)

            print(f"Accuracy: {test_acc:.4f} | Tiempo: {fin - ini:.2f}s")

        acc_media = np.mean(aciertos)
        tiempo_medio = np.mean(tiempos)

        resultados.append((acc_media, tiempo_medio))

        print(f"Accuracy media: {acc_media:.4f}")
        print(f"Tiempo medio: {tiempo_medio:.2f}s")

    plot_Barras(
        resultados,
        etiquetas,
        f_parametro="tamaños del kernel",
        titulo="Accuracy y tiempo con diferentes kernels"
    )

    
def main():
    
    #X_train, X_test, Y_train, Y_test = cargar_y_preprocesar_cifar10()
    #tarea_1(X_train, X_test, Y_train, Y_test)
    #tarea_2(X_train, X_test, Y_train, Y_test)
    #tarea_3(X_train, X_test, Y_train, Y_test)
    #tarea_4(X_train, X_test, Y_train, Y_test)
    #tarea_5(X_train, X_test, Y_train, Y_test)
    '''
    arquitecturas2 = ["700x300","600x400","400x600","300x700","500x500"]
    arquitecturas3 = ["501x166x333","501x333x166","166x333x501","333x501x166","334x333x333"]
    arquitecturas4 = ["300x200x200x300","300x300x200x200","200x200x300x300","200x300x300x200","250x250x250x250"]
    arquitecturas5 = ["273x182x90x182x273","300x100x200x100x300","90x182x273x182x90","90x90x182x182x273","200x200x200x200x200"]
    
    tarea_6(X_train, X_test, Y_train, Y_test)
    '''
    #tarea_7(X_train, X_test, Y_train, Y_test)
    X_train, X_test, Y_train, Y_test = cargar_y_preprocesar_cifar10_cnn()
    #cnn1(X_train, X_test, Y_train, Y_test)
    cnn2(X_train, X_test, Y_train, Y_test)
    
    

if __name__ == "__main__": 
    main()