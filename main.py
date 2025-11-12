'''
Eliminar los mensajes de error por no instalar el soporte de CUDA
'''
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import matplotlib.pyplot as plt
from random import sample

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

def main():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    print(X_train.shape, X_train.dtype) #  array de 50 000 imágenes en color de 32x32 píxeles
    print(Y_train.shape, Y_train.dtype) #  50 000 números asociados a cada categoría (0: avión; 1: coche; 2: pájaro, etc.)
    print(X_test.shape, X_test.dtype) # array de 10 000 imágenes en color de 32x32 píxeles
    print(Y_test.shape, Y_test.dtype)
 # 10 000 números asociados a cada categoría (0: avión; 1: coche; 2: pájaro, etc.)
    
    for i in sample(list(range(len(X_train))), 3):
        titulo = "Mostrando imagen X_train[" + str(i) + "]"
        titulo = titulo + " -- Y_train[" + str(i) + "] = " + str(Y_train[i])
        show_image(X_train[i], titulo)
        
    plot_curva(Y_test[:20], "Etiquetas de los primeros 20 valores")
    
    
if __name__ == "__main__": main()