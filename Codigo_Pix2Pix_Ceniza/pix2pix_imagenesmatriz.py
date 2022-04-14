
# ejemplo de carga de un modelo pix2pix y su uso para la traducción de imágenes puntuales
from os import listdir
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from numpy import vstack
import os
 
# Cargar una imagen
def load_image(filename, size=(256,256)):
	# Cargar la imagen con el tamaño preferido
	pixels = load_img(filename, target_size=size)
	# Convertir en matriz numpy
	pixels = img_to_array(pixels)
	# Escala de [0,255] a [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# Reajustar a 1 muestra
	pixels = expand_dims(pixels, 0)
	return pixels
 
# Cargar la imagen de origen
path = 'ceniza/test'
titles = ['Source', 'Generated']
        
for filename in listdir(path):
        os.chdir('D:/UniversidadCentraldelEcuador/NovenoSemestre/SoftwareAplicadoalaGeologia/ProyectoGANs/GANs_Ceniza/pix2pix')
        # Carga la imagen original
        src_image = load_image(path + '/' + filename)
        print('Loaded', src_image.shape)
        # Carga el Modelo
        model = load_model('Modelos/model_20.h5')
        # Genera la imagen desde la fuente 
        gen_image = model.predict(src_image)
        images = vstack((src_image, gen_image))
        # Escala de [-1,1] a [0,1]
        images = (images + 1) / 2.0
        # Trazar la imagen
        for i in range(len(images)):
            pyplot.subplot(1, 2, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(images[i])
            pyplot.title(titles[i])
        
        # Nombre con el que se guarda la imagen
        final = len(filename) -4 
        print(final)
        filename1 = 'plot_'+filename[0:len(filename)-4:1]+'.png'
        os.chdir('test_model_20')
        # Guarda la imagen en el computador
        pyplot.savefig(filename1)

os.chdir('D:/UniversidadCentraldelEcuador/NovenoSemestre/SoftwareAplicadoalaGeologia/ProyectoGANs/GANs_Ceniza/pix2pix')
for filename in listdir(path):    
        os.chdir('D:/UniversidadCentraldelEcuador/NovenoSemestre/SoftwareAplicadoalaGeologia/ProyectoGANs/GANs_Ceniza/pix2pix')
        # Carga la imagen original
        src_image = load_image(path + '/' + filename)
        print('Loaded', src_image.shape)
        # Carga el Modelo
        model = load_model('Modelos/model_30.h5')
        # Genera la imagen desde la fuente 
        gen_image = model.predict(src_image)
        images = vstack((src_image, gen_image))
        # Escala de [-1,1] a [0,1]
        images = (images + 1) / 2.0
        for i in range(len(images)):
            pyplot.subplot(1, 2, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(images[i])
            pyplot.title(titles[i])

        # Nombre con el que se guarda la imagen
        final = len(filename) -4
        print(final)
        filename1 = 'plot_'+filename[0:len(filename)-4:1]+'.png'
        os.chdir('test_model_30')
        # Guarda la imagen en el computador
        pyplot.savefig(filename1)
  

os.chdir('D:/UniversidadCentraldelEcuador/NovenoSemestre/SoftwareAplicadoalaGeologia/ProyectoGANs/GANs_Ceniza/pix2pix')
for filename in listdir(path):    
        os.chdir('D:/UniversidadCentraldelEcuador/NovenoSemestre/SoftwareAplicadoalaGeologia/ProyectoGANs/GANs_Ceniza/pix2pix')
        # Carga la imagen original
        src_image = load_image(path + '/' + filename)
        print('Loaded', src_image.shape)
        # Carga el Modelo
        model = load_model('Modelos/model_50.h5')
       # Genera la imagen desde la fuente 
        gen_image = model.predict(src_image)
        gen_image = (gen_image + 1) / 2.0
        images = vstack((src_image, gen_image))
        # Escala de [-1,1] a [0,1]
        images = (images + 1) / 2.0
        # Trazar la imagen
        for i in range(len(images)):
            pyplot.subplot(1, 2, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(images[i])
            pyplot.title(titles[i])

        # Nombre con el que se guarda la imagen    
        final = len(filename) -4
        print(final)
        filename1 = 'plot_'+filename[0:len(filename)-4:1]+'.png'
        os.chdir('test_model_50')
        # Guarda la imagen en el computador
        pyplot.savefig(filename1)
  
os.chdir('D:/UniversidadCentraldelEcuador/NovenoSemestre/SoftwareAplicadoalaGeologia/ProyectoGANs/GANs_Ceniza/pix2pix')
for filename in listdir(path):    
        os.chdir('D:/UniversidadCentraldelEcuador/NovenoSemestre/SoftwareAplicadoalaGeologia/ProyectoGANs/GANs_Ceniza/pix2pix')
        # Carga la imagen original
        src_image = load_image(path + '/' + filename)
        print('Loaded', src_image.shape)
        # Carga el Modelo
        model = load_model('Modelos/model_90.h5')
       # Genera la imagen desde la fuente 
        gen_image = model.predict(src_image)
        gen_image = (gen_image + 1) / 2.0
        images = vstack((src_image, gen_image))
        # Escala de [-1,1] a [0,1]
        images = (images + 1) / 2.0
        # Trazar la imagen
        for i in range(len(images)):
            pyplot.subplot(1, 2, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(images[i])
            pyplot.title(titles[i])

        # Nombre con el que se guarda la imagen    
        final = len(filename) -4
        print(final)
        filename1 = 'plot_'+filename[0:len(filename)-4:1]+'.png'
        os.chdir('test_model_90')
        # Guarda la imagen en el computador
        pyplot.savefig(filename1)
  
pyplot.close()