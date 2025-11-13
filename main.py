'''
Eliminar los mensajes de error por no instalar el soporte de CUDA
'''
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
from keras import layers
from keras import ops
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from keras.utils import to_categorical

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

'''
def plot_Historys(Historys):
    
    train_accuracy = []
    train_loss = []
    validation_accuracy = []
    validation_loss = []
        
    for h in Historys:
        train_accuracy.append(h.history['accuracy'])
        train_loss.append(h.history['loss'])
        validation_accuracy.append(h.history['val_accuracy'])
        validation_loss.append(h.history['val_loss'])

    # ====== PLOT ACCURACY ======
    plt.figure(figsize=(12,5))

    for acc, val_acc in zip(train_accuracy, validation_accuracy):
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'b-', alpha=0.4)
        plt.plot(epochs, val_acc, 'r--', alpha=0.4)

        
    plt.title("Curvas de accuracy (cada entrenamiento)")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend(["Entrenamiento", "Validación"])

    # ====== PLOT LOSS ======
    plt.subplot(1,2,2)
    for loss, val_loss in zip(train_loss, validation_loss):
        plt.plot(epochs, loss, 'b-', alpha=0.4)
        plt.plot(epochs, val_loss, 'r--', alpha=0.4)
    plt.title("Curvas de loss (cada entrenamiento)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend(["Entrenamiento", "Validación"])

    plt.tight_layout()
    plt.show()
'''     

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
    
    my_callbacks = [
        keras.callbacks.EarlyStopping(patience=2),
        keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.keras'),
        keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    
    Historys = []
    for i in range(5):
        print(f"Entrenamiento numero {i+1}")
        Historys.append(model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.1,callbacks=my_callbacks))
        test_scores = model.evaluate(X_test, Y_test, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])
    
    plot_Historys(Historys)
    
if __name__ == "__main__": main()