import string

import torch


def split(word: str) -> list:
    return [char for char in word]

blank_character = '<BLANK>'
void_character = '<NOTHING>'
#characters = [blank_character, void_character, 'a', 'b']
characters = [blank_character, void_character] + split(string.ascii_letters) + split(string.digits) + [" ", ",", '"', "'", "_", "-", "&", "."]
nb_characters = len(characters)

blank_id = characters.index(blank_character)
nothing_id = characters.index(void_character)
pad_id = len(characters)


def char_index(char: chr) -> int:
    try:
        return characters.index(char)
    except ValueError:
        print(f"Could not find character {char}")
        return 0


def index_char(idx):
    if len(characters) > idx:
        return characters[idx]
    return 'Ø'


def sentence_to_list(sentence: str, padding: int = 100) -> list:
    return ([char_index(char) for char in sentence] + padding * [pad_id])[:padding]
    #return [char_index(char) for char in sentence]


def get_sentence_length(sentence: torch.Tensor) -> int:
    return (sentence == pad_id).nonzero(as_tuple=True)[0][0].item()

def get_sentence_length_test(sentence: torch.Tensor) -> int:
    return (sentence == pad_id).nonzero()[0][0].item()




def get_selected_character(i: torch.Tensor):
    return torch.argmax(i).item()


def from_target_labels(target: torch.Tensor) -> str:
    """

    :param target: tensor of shape (n) with n being the length of the sequence
    and each element containing the index of one of the character
    :return: a trimmed string containing only relevant characters
    """
    target_array = target.cpu().numpy().astype(int)
    target_array = target_array[:get_sentence_length_test(target_array)]
    return ''.join([characters[i] for i in target_array])


def from_predicted_labels(predicted: torch.Tensor) -> str:
    """

    :param predicted: tensor of shape (L, X) with :
    - L being the length of the sequence
    - X being the size of the list of known characters
    and each element containing the index of one of the character
    :return: a trimmed string containing only relevant characters
    """
    as_np = predicted.detach()
    all_chars = [get_selected_character(i) for _, i in enumerate(as_np)]
    final = []
    current_char = None
    for char in all_chars:
        if not char == current_char:
            current_char = char
            if char == 0:
                pass
            else:
                final.append(char)
    return ''.join([str(characters[i]) for i in final])