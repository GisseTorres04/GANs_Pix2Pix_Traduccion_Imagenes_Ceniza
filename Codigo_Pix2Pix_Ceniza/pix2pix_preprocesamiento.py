
import numpy as np
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.initializers import RandomNormal
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization


class Pix2PixPreprocesamiento():
    def _init_(self):
        super().__init__()

    
    # Definir un bloque codificador
    def definir_bloque_codificador(self,layer_in, n_filters, batchnorm=True):
	    # Inicialización de los pesos
        init = RandomNormal(stddev=0.02)
	    # Se añade capa downsampling
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	    # Se añade normalización por lotes
        if batchnorm:
            g = BatchNormalization()(g, training=True)
	    # Añadir capa de función de activación
        g = LeakyReLU(alpha=0.2)(g)
        return g
 
 
    # Definir un bloque decodificador
    def definir_bloque_decodificador(self,layer_in, skip_in, n_filters, dropout=True):
	    # Inicialización de pesos
        init = RandomNormal(stddev=0.02)
	    # Se añade capa de upsampling
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        # Se añade normalización por lotes
        g = BatchNormalization()(g, training=True)
	    # Añadir capa condición dropout
        if dropout:
            g = Dropout(0.5)(g, training=True)
	    # Fusionar con la conexión de salto
        g = Concatenate()([g, skip_in])
        # Añadir capa de función de activación
        g = Activation('relu')(g)
        return g
 
 
    # Cargar y preparar las imágenes de entrenamiento
    def cargar_ejemplos_reales(self,filename):
	    # Cargar matrices comprimidas
        data = load(filename)
	    # Desempacar arreglos
        X, Y = data['arr_0'], data['arr_1']
	    # Escala de [0,255] a [-1,1] - Normalización
        X = (X - 127.5) / 127.5
        Y = (Y - 127.5) / 127.5
        return [X, Y]
 
    # Selecciona un lote de muestras aleatorias, devuelve las imágenes y el objetivo
    def generar_ejemplos_reales(self,dataset, n_samples, patch_shape):
	    # Descomprimir dataset
        X, Y = dataset
	    # Elegir una instancia al azar
        ix = randint(0, X.shape[0], n_samples)
	    # Recuperar las imágenes seleccionadas
        delta_x, delta_y = X[ix], Y[ix]
	    # Generar etiquetas de clase "reales" (1)
        y = ones((n_samples, patch_shape, patch_shape, 1))
        return [delta_x, delta_y], y
 
 
    # Genera un lote de imágenes, devuelve objetos e imágenes
    def generar_ejemplos_falsos(self,g_model, samples, patch_shape):
    # Generar una instancia falsa
        X = g_model.predict(samples)
	    # Crear etiquetas de clase "falsas" (0)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        return X, y
 