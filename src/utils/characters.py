import string


def split(word: str) -> list:
    return [char for char in word]

blank_character = '<BLANK>'
void_character = '<NOTHING>'
characters = [blank_character, void_character] + split(string.ascii_letters) + split(string.digits) + [" ", ",", '"', "'"]

blank_id = characters.index(blank_character)
nothing_id = characters.index(void_character)


def char_index(char: chr) -> int:
    try:
        return characters.index(char)
    except ValueError:
        return 0


def index_char(idx):
    if len(characters) > idx:
        return characters[idx]
    return 'Ø'


def sentence_to_list(sentence: str, padding: int=100) -> list:
    return ([char_index(char) for char in sentence] + padding * [nothing_id])[:padding]

