# ejemplo de carga de un modelo pix2pix y su uso para la traducción de imágenes puntuales
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from numpy import vstack

#Cargar una imagen
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


titles = ['Source', 'Generated']
# Carga la imagen original
src_image = load_image('Imagenes_de_Prueba/Volcan_Cumbre_Vieja.jpg')
print('Loaded', src_image.shape)
# Carga el Modelo
model = load_model('Modelos/model_90.h5')
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
# Guarda la imagen en el computador
pyplot.savefig('Imagenes_Predecidas/Volcan_Cumbre_Vieja.jpg')


