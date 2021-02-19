from utils.image_helper import TextGenerator
import sys
path = sys.argv[1]
nb_images = sys.argv[2]
nb_characters = sys.argv[3]
height = sys.argv[4]
width = sys.argv[5]
tg = TextGenerator()
tg.generate(path, nb_images=nb_images, nb_characters=nb_characters, height=height, width=width)