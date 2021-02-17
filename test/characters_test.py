import sys
sys.path.insert(1, '/opt/projetcs/ich/e2eocr/src')
from utils.characters import sentence_to_list

array = sentence_to_list("Coucou c'est moi")
assert array == \
       ([30, 16, 22, 4, 16, 22, 64, 4, 67, 6, 20, 21, 64, 14, 16, 10] + 100 * [1])[:100]