#!$CONDA_PREFIX/bin/python3

####################
# Required modules #
####################

# Generic


# Libs


# Custom
from GENRE.genre.fairseq_model import mGENREHubInterface
from GENRE.genre.trie import MarisaTrie

##################
# Configurations #
##################

#############
# Functions #
#############


def inference_with_mgenre(
    sentences: list[str],
    model: mGENREHubInterface,
    knowledge_base: dict[tuple, set],
    candidate_trie: MarisaTrie,
) -> list[list[dict]]:
    """Use mGENRE to infer the most likely entities for a given list of sentences.

    Args:
        sentences (list[str]): List of sentences to infer entities for.
        model (mGENREHubInterface): mGENRE model.
        knowledge_base (dict[tuple, set]): Knowledge base.
            Format: {(language, title): {set of QIDs}, ...}
        candidate_trie (MarisaTrie): Trie of candidate entities.
            See GENRE.genre.trie for more details.

    Returns:
        list[list[dict]]: List of entities for each sentence.
            Format: [[{ 'id': QID,
                        'texts': [f'{page title} >> {language}'],
                        'scores': tensor([scores]), # See GENRE paper for details on scoring
                        'score': tensor(score)
                      },
                      {option 2},...
                     ], Results for sentence 2, ...
                    ]
    """
    # See examples_mgenre/examples.ipynb for details on how to use mGENRE
    return model.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e
            for e in candidate_trie.get(sent.tolist())
            if e < len(model.task.target_dictionary)
        ],
        text_to_id=lambda x: max(
            knowledge_base[tuple(reversed(x.split(" >> ")))], key=lambda y: int(y[1:])
        ),
        marginalize=True,
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
