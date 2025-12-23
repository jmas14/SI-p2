#Jesús Ballesteros Navarro 50387249E
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
#import matplotlib
#matplotlib.use('Agg')  # ← Modo no-interactivo
import matplotlib.pyplot as plt
import keras
from keras import layers, models, callbacks
import time
###################################################
#           Funciones auxiliares
###################################################

def cargar_y_preprocesar_cifar10():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    
    # Normalizar
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    # Aplanar para MLP
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # One-hot encoding
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)
    
    return X_train, Y_train, X_test, Y_test 

def promediar_histories(all_histories):
    max_len = max(len(h['loss']) for h in all_histories)
    
    avg_history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(max_len):
        losses = [h['loss'][epoch] for h in all_histories if epoch < len(h['loss'])]
        accs = [h['accuracy'][epoch] for h in all_histories if epoch < len(h['accuracy'])]
        val_losses = [h['val_loss'][epoch] for h in all_histories if epoch < len(h['val_loss'])]
        val_accs = [h['val_accuracy'][epoch] for h in all_histories if epoch < len(h['val_accuracy'])]
        
        if losses:
            avg_history['loss'].append(np.mean(losses))
            avg_history['accuracy'].append(np.mean(accs))
            avg_history['val_loss'].append(np.mean(val_losses))
            avg_history['val_accuracy'].append(np.mean(val_accs))
    
    return avg_history

##################################################
#                   Graficas
##################################################
def plot_history(history): #Función para plotear grafica lineal

    hist = history.history if hasattr(history, 'history') else history #Permitir tanto objetos history como diccionarios

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de Accuracy
    ax1.plot(hist['accuracy'], label='Train Accuracy', linewidth=2)  
    ax1.plot(hist['val_accuracy'], label='Val Accuracy', linewidth=2)  
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Evolución de Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #Gráfica de Loss
    ax2.plot(hist['loss'], label='Train Loss', linewidth=2)  
    ax2.plot(hist['val_loss'], label='Val Loss', linewidth=2)  
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.set_title('Evolución de Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('history.png', dpi=150) 
    plt.close(fig)


def plot_confusion_matrix(Y_true, Y_pred):  
    clases = ['avión', 'coche', 'pájaro', 'gato', 'ciervo',
              'perro', 'rana', 'caballo', 'barco', 'camión']
    
    Y_true_idx = np.argmax(Y_true, axis=1)
    Y_pred_idx = np.argmax(Y_pred, axis=1)
    
    n_clases = 10
    cm = np.zeros((n_clases, n_clases), dtype=int)
    
    for true, pred in zip(Y_true_idx, Y_pred_idx):
        cm[true][pred] += 1
    
    # Plotear
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    
    # Etiquetas
    ax.set_xticks(np.arange(n_clases))
    ax.set_yticks(np.arange(n_clases))
    ax.set_xticklabels(clases, rotation=45, ha='right')
    ax.set_yticklabels(clases)
    
    # Números en celdas
    for i in range(n_clases):
        for j in range(n_clases):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    
    ax.set_title('Matriz de Confusión')
    ax.set_ylabel('Etiqueta Original')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('confusion.png', dpi=150)
    plt.close(fig)

def plot_comparacion(resultados, param_name, titulo, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Comparación de configuraciones de {titulo}', fontsize=16)
    
    # Extraer valores
    param_vals = [r[param_name] for r in resultados]
    epochs_vals = [r['epochs'] for r in resultados]
    acc_vals = [r['test_acc'] for r in resultados]
    tiempo_vals = [r['tiempo'] for r in resultados]
    
    # Subplot 1: Epochs promedio
    axes[0].bar(range(len(param_vals)), epochs_vals, color='steelblue')
    axes[0].set_xticks(range(len(param_vals)))
    axes[0].set_xticklabels([str(p) for p in param_vals], rotation=15 if len(str(param_vals[0])) > 5 else 0)
    axes[0].set_xlabel(titulo)
    axes[0].set_ylabel('Epochs promedio')
    axes[0].set_title('Número de epochs entrenados')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(epochs_vals):
        axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Test accuracy
    axes[1].bar(range(len(param_vals)), acc_vals, color='forestgreen')
    axes[1].set_xticks(range(len(param_vals)))
    axes[1].set_xticklabels([str(p) for p in param_vals], rotation=15 if len(str(param_vals[0])) > 5 else 0)
    axes[1].set_xlabel(titulo)
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Precisión en test')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(acc_vals):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Tiempo
    axes[2].bar(range(len(param_vals)), tiempo_vals, color='coral')
    axes[2].set_xticks(range(len(param_vals)))
    axes[2].set_xticklabels([str(p) for p in param_vals], rotation=15 if len(str(param_vals[0])) > 5 else 0)
    axes[2].set_xlabel(titulo)
    axes[2].set_ylabel('Tiempo (segundos)')
    axes[2].set_title('Tiempo de entrenamiento')
    axes[2].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(tiempo_vals):
        axes[2].text(i, v + 1, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def entrenamiento_ES(X_train, Y_train, X_test, Y_test, patience=5, max_epochs=100, num_repeticiones=5):
    test_accs = []  #Arrays para guardar los datos del entreno
    test_losses = []
    epochs_trained = []
    tiempos = []
    all_histories = [] 
    
    for rep in range(num_repeticiones):
        
        model = keras.Sequential([
            layers.Dense(48, activation="sigmoid", input_shape=(3072,)),
            layers.Dense(10, activation="softmax")
        ])

        model.summary()
        
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Configurar EarlyStopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',          
            patience=patience,             
            restore_best_weights=True,    
            verbose=0
        )
        
        # Entrenar
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            batch_size=32,
            epochs=max_epochs,             
            validation_split=0.1,
            callbacks=[early_stop],       
            verbose=1
        )

        tiempo = time.time() - inicio
        
    
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0) 
        
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        epochs_trained.append(len(history.history['loss']))  #Saber los epochs realmente entrenados
        tiempos.append(tiempo)
        all_histories.append(history.history)

    # Calcular promedios
    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)
    avg_epochs = np.mean(epochs_trained)
    avg_tiempo = np.mean(tiempos)
    avg_history = promediar_histories(all_histories)
    
    return {
        'patience': patience,
        'test_acc': avg_test_acc,
        'test_loss': avg_test_loss,
        'epochs': avg_epochs,
        'tiempo': avg_tiempo,
        'history': avg_history
    }

def entrenamiento_batch_size(X_train, Y_train, X_test, Y_test, batch_size, patience=5, max_epochs=100, num_repeticiones=5): #Entrenamiento basado en los batchsizes, mlp3
    test_accs = []
    test_losses = []
    epochs_trained = []
    tiempos = []
    all_histories = []
    
    for rep in range(num_repeticiones):
        print(f"  Repetición {rep+1}/{num_repeticiones}...", end=" ")
        
        model = keras.Sequential([
            layers.Dense(48, activation="sigmoid", input_shape=(3072,)),
            layers.Dense(10, activation="softmax")
        ])

        model.summary()
        
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        tiempo = time.time() - inicio
        
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        epochs_trained.append(len(history.history['loss']))
        tiempos.append(tiempo)
        all_histories.append(history.history)
        
       
    # Promedios
    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)
    avg_epochs = np.mean(epochs_trained)
    avg_tiempo = np.mean(tiempos)
    
    avg_history = promediar_histories(all_histories)
    
    return {
        'batch_size': batch_size,
        'test_acc': avg_test_acc,
        'test_loss': avg_test_loss,
        'epochs': avg_epochs,
        'tiempo': avg_tiempo,
        'history': avg_history
    }

def entrenamiento_activation(X_train, Y_train, X_test, Y_test, activation, patience=5, batch_size=32, max_epochs=100, num_repeticiones=5): #Entrenamiento variando el modo de activación
    test_accs = []
    test_losses = []
    epochs_trained = []
    tiempos = []
    all_histories = []
    
    for rep in range(num_repeticiones):
        model = keras.Sequential([
            layers.Dense(48, activation=activation, input_shape=(3072,)),  # ← ACTIVACIÓN VARIABLE
            layers.Dense(10, activation="softmax")
        ])
        
        model.summary()

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        tiempo = time.time() - inicio
        
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        epochs_trained.append(len(history.history['loss']))
        tiempos.append(tiempo)
        all_histories.append(history.history)
        
        
    
    # Promedios
    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)
    avg_epochs = np.mean(epochs_trained)
    avg_tiempo = np.mean(tiempos)
    
    avg_history = promediar_histories(all_histories)
    
    return {
        'activation': activation,
        'test_acc': avg_test_acc,
        'test_loss': avg_test_loss,
        'epochs': avg_epochs,
        'tiempo': avg_tiempo,
        'history': avg_history
    }


def entrenamiento_neurons(X_train, Y_train, X_test, Y_test, neurons, activation='elu', patience=5, batch_size=128, max_epochs=100, num_repeticiones=5): #Entrenamiento de neuronas con los parametros optimos anteriores para mlp5
    test_accs = []
    test_losses = []
    epochs_trained = []
    tiempos = []
    all_histories = []
    
    for rep in range(num_repeticiones):
        
        model = keras.Sequential([
            layers.Dense(neurons, activation=activation, input_shape=(3072,)), 
            layers.Dense(10, activation="softmax")
        ])
        
        model.summary()

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        tiempo = time.time() - inicio
        
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        epochs_trained.append(len(history.history['loss']))
        tiempos.append(tiempo)
        all_histories.append(history.history)
          
    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)
    avg_epochs = np.mean(epochs_trained)
    avg_tiempo = np.mean(tiempos)
    avg_history = promediar_histories(all_histories)
    
    return {
        'neurons': neurons,
        'test_acc': avg_test_acc,
        'test_loss': avg_test_loss,
        'epochs': avg_epochs,
        'tiempo': avg_tiempo,
        'history': avg_history
    }

def entrenamiento_arquitectura(X_train, Y_train, X_test, Y_test, arquitectura, arquitectura_nombre, activation='relu', patience=5, batch_size=32, max_epochs=100, num_repeticiones=5): #Entrenamiento por capas con lo optimizado anteriormente, mlp6
    test_accs = []
    test_losses = []
    epochs_trained = []
    tiempos = []
    all_histories = []
    
    for rep in range(num_repeticiones):
        # Crear modelo con arquitectura parametrizada
        model = keras.Sequential()
        
        # Primera capa
        model.add(layers.Dense(arquitectura[0], activation=activation, input_shape=(3072,)))
        
        # Capas ocultas adicionales
        for neuronas in arquitectura[1:]:
            model.add(layers.Dense(neuronas, activation=activation))
        
        # Capa de salida
        model.add(layers.Dense(10, activation="softmax"))
        
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        tiempo = time.time() - inicio
        
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        epochs_trained.append(len(history.history['loss']))
        tiempos.append(tiempo)
        all_histories.append(history.history)

    # Promedios
    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)
    avg_epochs = np.mean(epochs_trained)
    avg_tiempo = np.mean(tiempos)
    
    avg_history = promediar_histories(all_histories)
    
    return {
        'arquitectura': arquitectura_nombre,
        'arquitectura_list': arquitectura,
        'test_acc': avg_test_acc,
        'test_loss': avg_test_loss,
        'epochs': avg_epochs,
        'tiempo': avg_tiempo,
        'history': avg_history
    }


################################################################
#                            MLP
################################################################
def mlp1():
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    model = models.Sequential()

    model = keras.Sequential([
        layers.Dense(48, activation="sigmoid", input_shape=(3072,)),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        X_train, Y_train,
        validation_split=0.1,
        batch_size=32,
        epochs=10
    )

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    plot_history(history)
    Y_pred = model.predict(X_test)
    plot_confusion_matrix(Y_test, Y_pred)

    return model, history

def mlp2():
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    
    patience_values = [3, 5, 7, 10, 15]
    resultados = []
    
    for patience in patience_values:
        resultado = entrenamiento_ES(
            X_train, Y_train, X_test, Y_test,
            patience=patience,
            max_epochs=100,
            num_repeticiones=5
        )
        resultados.append(resultado)

   
    # Mejor configuración
    mejor = max(resultados, key=lambda x: x['test_acc'])
    plot_comparacion(resultados, 'patience', 'Patience (EarlyStopping)', 'mlp2_comparacion_patience.png')
    
    mejor_history = mejor['history']
    plot_history(mejor_history)
    
    return resultados, mejor

def mlp3():
    batch_sizes = [16, 32, 64, 128, 256]
    patience = 5

    
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    
    batch_sizes = [16, 32, 64, 128, 256]
    mejor_patience = 5  
    
    resultados = []
    
    for bs in batch_sizes:
        resultado = entrenamiento_batch_size(
            X_train, Y_train, X_test, Y_test,
            batch_size=bs,
            patience=mejor_patience,
            num_repeticiones=5
        )
        resultados.append(resultado)
    
    # Mejor configuración
    mejor = max(resultados, key=lambda x: x['test_acc'])
    
    # Gráficas
    plot_comparacion(resultados, 'batch_size', 'Batch Size', 'mlp3_comparacion_batch_size.png')
    plot_history(mejor['history'])
    
    return resultados, mejor


def mlp4():
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    
    activations = ['sigmoid', 'relu', 'tanh', 'elu']
    mejor_patience = 5     
    mejor_batch_size = 128
    
    resultados = []
    
    for act in activations:
        resultado = entrenamiento_activation(
            X_train, Y_train, X_test, Y_test,
            activation=act,
            patience=mejor_patience,
            batch_size=mejor_batch_size,
            num_repeticiones=5
        )
        resultados.append(resultado)
    
    # Mejor configuración
    mejor = max(resultados, key=lambda x: x['test_acc'])
    
    # Gráficas
    plot_comparacion(resultados, 'activation', 'Función de Activación', 'mlp4_comparacion_activation.png')
    plot_history(mejor['history'])
    
    return resultados, mejor


def mlp5():
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    
    neurons_list = [24, 48, 96, 128, 256]
    mejor_patience = 5
    mejor_batch_size = 128
    mejor_activation = 'elu'
    
    resultados = []
    
    for n in neurons_list:
        resultado = entrenamiento_neurons(
            X_train, Y_train, X_test, Y_test,
            neurons=n,
            activation=mejor_activation,
            patience=mejor_patience,
            batch_size=mejor_batch_size,
            num_repeticiones=5
        )
        resultados.append(resultado)
    # Mejor configuración
    mejor = max(resultados, key=lambda x: x['test_acc'])
    
    # Gráficas
    plot_comparacion(resultados, 'neurons', 'Número de Neuronas', 'mlp5_comparacion_neurons.png')
    plot_history(mejor['history'])
    
    return resultados, mejor

def mlp6():  
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    
    # Arquitecturas a probar
    # Formato: (lista_neuronas, nombre_descriptivo)
    arquitecturas = [
    # 1 CAPA OCULTA (comparar tamaños)
        ([128], "1 capa: 128"),
        ([256], "1 capa: 256"),
        
        # 2 CAPAS OCULTAS (diferentes distribuciones)
        ([128, 64], "2 capas: 128-64"),
        ([256, 128], "2 capas: 256-128"),
        ([128, 128], "2 capas: 128-128"),
        
        # 3 CAPAS OCULTAS (explorar profundidad)
        ([128, 64, 32], "3 capas: 128-64-32"),
        ([256, 128, 64], "3 capas: 256-128-64"),
        ([96, 96, 96], "3 capas: 96-96-96")
    ]
    
    mejor_patience = 5         
    mejor_batch_size = 128     
    mejor_activation = 'elu'   
    
    resultados = []
    
    for arq_list, arq_nombre in arquitecturas:
        resultado = entrenamiento_arquitectura(
            X_train, Y_train, X_test, Y_test,
            arquitectura=arq_list,
            arquitectura_nombre=arq_nombre,
            activation=mejor_activation,
            patience=mejor_patience,
            batch_size=mejor_batch_size,
            num_repeticiones=5
        )
        resultados.append(resultado)

    # Mejor configuración
    mejor = max(resultados, key=lambda x: x['test_acc'])
    
    # Gráficas
    plot_comparacion(resultados, 'arquitectura', 'Arquitectura', 'mlp6_comparacion_arquitectura.png')
    plot_history(mejor['history'])
    
    return resultados, mejor

def mlp7():
    mejor_patience = 5              
    mejor_batch_size = 128          
    mejor_activation = 'elu'        
    mejor_arquitectura = [256, 128, 64]
 
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    num_repeticiones = 5
    
    test_accs = []
    test_losses = []
    epochs_trained = []
    tiempos = []
    all_histories = []
    
    for rep in range(num_repeticiones):
        model = keras.Sequential()
        
        # Primera capa
        model.add(layers.Dense(mejor_arquitectura[0], activation=mejor_activation, input_shape=(3072,)))
        
        # Capas ocultas adicionales
        for n in mejor_arquitectura[1:]:
            model.add(layers.Dense(n, activation=mejor_activation))
        
        # Capa de salida
        model.add(layers.Dense(10, activation="softmax"))
        
        model.summary()

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=mejor_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        inicio = time.time()
        history = model.fit(
            X_train, Y_train,
            batch_size=mejor_batch_size,
            epochs=100,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        tiempo = time.time() - inicio
        
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        epochs_trained.append(len(history.history['loss']))
        tiempos.append(tiempo)
        all_histories.append(history.history)
 

    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)
    avg_epochs = np.mean(epochs_trained)
    avg_tiempo = np.mean(tiempos)
    avg_history = promediar_histories(all_histories)
    

    mlp1_acc = 0.4219    
    mlp1_tiempo = 47.0      
    mlp1_epochs = 10       
    mlp1_loss = 1.6169 
    
    resultados_comparacion = [
        {
            'modelo': 'MLP1\n(Baseline)',
            'test_acc': mlp1_acc,
            'test_loss': mlp1_loss,
            'epochs': mlp1_epochs,
            'tiempo': mlp1_tiempo
        },
        {
            'modelo': 'MLP7\n(Optimizado)',
            'test_acc': avg_test_acc,
            'test_loss': avg_test_loss,
            'epochs': avg_epochs,
            'tiempo': avg_tiempo
        }
    ]
    
    #Gráficas
    plot_history(avg_history)
    plot_comparacion(resultados_comparacion, 'modelo', 'Modelo', 'mlp7_comparacion_baseline.png')
    
    Y_pred = model.predict(X_test, verbose=0)
    plot_confusion_matrix(Y_test, Y_pred)

#main
if __name__ == "__main__":
    cargar_y_preprocesar_cifar10()

    #mlp1()
    #mlp2()
    #mlp3()
    #mlp4()
    #mlp5()
    #mlp6()
    mlp7() #No se ha terminado de optimizar completamente el modelo del mlp7 aun falta probar más ajustes, sin embargo esta versión presenta una optimización considerable frente a mlp1
