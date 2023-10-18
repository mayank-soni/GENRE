#!$CONDA_PREFIX/bin/python3

####################
# Required modules #
####################

# Generic


# Libs
from tqdm import tqdm


# Custom
from config import (
    MGENRE_MARISA_TRIE_PATH,
    MGENRE_KNOWLEDGE_BASE_PATH,
    MGENRE_MODEL_PATH,
)
from GENRE.genre.fairseq_model import mGENRE, mGENREHubInterface
from GENRE.genre.trie import MarisaTrie
from damuel.utils import save_to_pickle, load_from_pickle

##################
# Configurations #
##################

#############
# Functions #
#############


def create_marisa_trie_from_knowledge_base_mgenre(
    model: mGENREHubInterface, knowledge_base: dict[tuple, set]
) -> MarisaTrie:
    """Creates a marisa trie from a knowledge base for use with mGENRE.

    Args:
        model (mGENREHubInterface): mGENRE model
        knowledge_base (dict[tuple, set]): mGENRE formatted knowledge base
            Format: {(language, title): {set of QIDs}, ...}

    Returns:
        MarisaTrie: mGENRE formatted Marisa trie
    """
    # See temporary trie creation code in examples_mgenre/examples.ipynb
    return MarisaTrie(
        [
            [2] + model.encode(f"{name} >> {lang}")[1:].tolist()
            for lang, name in tqdm(knowledge_base.keys())
        ]
    )


###########
# Classes #
###########

#############################
# Example Class - Classname #
#############################


# class Classname:
#     """Class purpose

#     Attributes:

#     """

#     def __init__(self):
#         # General attributes

#         # Network attributes

#         # Data attributes

#         # Optimisation attributes

#         # Export attributes

#     ############
#     # Checkers #
#     ############

#     ###########
#     # Setters #
#     ###########

#     ###########
#     # Helpers #
#     ###########


#     ##################
#     # Core Functions #
#     ##################

##########
# Script #
##########
if __name__ == "__main__":
    kb = load_from_pickle(MGENRE_KNOWLEDGE_BASE_PATH)
    model = mGENRE.from_pretrained(MGENRE_MODEL_PATH).eval()
    save_to_pickle(
        create_marisa_trie_from_knowledge_base_mgenre(model, kb),
        MGENRE_MARISA_TRIE_PATH,
    )
#
