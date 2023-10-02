#!$CONDA_PREFIX/bin/python3

####################
# Required modules #
####################

# Generic
import argparse
import json
import pickle

# Libs
import evaluate

# Custom
from GENRE.genre.fairseq_model import mGENRE, mGENREHubInterface
from GENRE.genre.trie import MarisaTrie
from src.damuel.entity_linking_dataset import (
    extract_single_data_line,
    batch_generator_for_iterables,
    extract_single_mention_mgenre_format,
)

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


def get_top_qids(mgenre_output: list[list[dict]]) -> list[str]:
    return [sentence[0]["id"] for sentence in mgenre_output]


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
        description="Runs DaMuEL dataset through mGENRE. Saves the output to a file. Calculates top-1 accuracy."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to DaMuEL dataset",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to mGENRE model",
    )
    parser.add_argument(
        "--knowledge_base_path",
        type=str,
        required=True,
        help="Path to knowledge base",
    )
    parser.add_argument(
        "--candidate_marisa_trie_path",
        type=str,
        required=True,
        help="Path to candidate entities Marisa trie",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size",
    )
    args = parser.parse_args()
    model = mGENRE.from_pretrained(args.model_path).eval()
    with open(args.knowledge_base_path, "rb") as file:
        knowledge_base = pickle.load(file)
    with open(args.candidate_marisa_trie_path, "rb") as file:
        candidate_trie = pickle.load(file)
    metric = evaluate.load("accuracy")
    with open(args.output_path, "w") as file:
        for (
            mention_ids,
            input_sentences,
            gold_output_qids,
        ) in batch_generator_for_iterables(
            batch_size=args.batch_size,
            data=extract_single_mention_mgenre_format(
                extract_single_data_line(args.dataset_path)
            ),
        ):
            output = inference_with_mgenre(
                sentences=input_sentences,
                model=model,
                knowledge_base=knowledge_base,
                candidate_trie=candidate_trie,
            )
            for mention_id, input_sentence, output_result in zip(
                mention_ids, input_sentences, output
            ):
                json_to_write = {
                    "mention_id": mention_id,
                    "input_sentence": input_sentence,
                    "output_result": output_result,
                }
                file.write(json.dumps(json_to_write) + "\n")
            metric.add_batch(
                references=gold_output_qids, predictions=get_top_qids(output)
            )
    print(metric.compute())
