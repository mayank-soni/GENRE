#!$CONDA_PREFIX/bin/python3

####################
# Required modules #
####################

# Generic
import argparse
import logging
import pathlib
import torch

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
    Skips any data that results in an <unk> token (i.e. 3)

    Args:
        model (mGENREHubInterface): mGENRE model
        knowledge_base (dict[tuple, set]): mGENRE formatted knowledge base
            Format: {(language, title): {set of QIDs}, ...}

    Returns:
        MarisaTrie: mGENRE formatted Marisa trie
    """
    # See temporary trie creation code in examples_mgenre/examples.ipynb
    trie_list = []
    for lang, name in tqdm(knowledge_base.keys()):
        encoded = model.encode(f"{name} >> {lang}")[1:].tolist()
        try:
            index = encoded.index(3)
        except ValueError:
            index = None
        if index:
            part_that_could_be_encoded = model.decode(torch.tensor(encoded[:index]))
            logging.warning(
                f"Knowledge base entry ({name} >> {lang}) contains token that cannot be encoded."
                + f" Part that could be encoded: {part_that_could_be_encoded}"
            )
            continue
        else:
            trie_list.append([2] + encoded)
    return MarisaTrie(trie_list)


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
    parser = argparse.ArgumentParser(
        description="Creates a marisa trie from a knowledge base for use with mGENRE."
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    parser.add_argument(
        "--logfile_path",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    output_path = (
        pathlib.Path(args.output_path) if args.output_path else MGENRE_MARISA_TRIE_PATH
    )
    logfile_path = pathlib.Path(args.logfile_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile_path)],
        force=True,
    )
    kb = load_from_pickle(MGENRE_KNOWLEDGE_BASE_PATH)
    model = mGENRE.from_pretrained(MGENRE_MODEL_PATH).eval()
    save_to_pickle(
        create_marisa_trie_from_knowledge_base_mgenre(model, kb),
        output_path,
    )
