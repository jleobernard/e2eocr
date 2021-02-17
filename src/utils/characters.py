import string


def split(word: str) -> list:
    return [char for char in word]


characters = ['<BLANK>', '<NOTHING>'] + split(string.ascii_letters) + split(string.digits) + [" ", ",", '"', "'"]

blank_id = characters.index('<BLANK>')
nothing_id = characters.index('<NOTHING>')

def char_index(char: chr) -> int:
    try:
        return characters.index(char)
    except ValueError:
        return 0


def sentence_to_list(sentence: str, padding: int=100) -> list:
    return ([char_index(char) for char in sentence] + padding * [nothing_id])[:padding]
