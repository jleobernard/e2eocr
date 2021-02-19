from utils.image_helper import TextGenerator
import sys
path = sys.argv[1]
nb_images = int(sys.argv[2])
nb_characters = int(sys.argv[3])
height = int(sys.argv[4])
width = int(sys.argv[5])
tg = TextGenerator()
tg.generate(path, nb_images=nb_images, nb_characters=nb_characters, height=height, width=width)