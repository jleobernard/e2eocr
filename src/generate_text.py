from utils.image_helper import TextGenerator

tg = TextGenerator()
tg.generate('../data/test/one-line', nb_images=10, nb_characters=5, height=80, width=80)