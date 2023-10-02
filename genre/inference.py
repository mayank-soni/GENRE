#!$CONDA_PREFIX/bin/python3

####################
# Required modules #
####################

# Generic
import argparse
from collections.abc import Generator
import json
import logging
import pickle

# Libs
import evaluate
from tqdm import tqdm

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

def get_json_serialisable_mgenre_output(mgenre_output: list[list[dict]]) -> list[list[dict]]:
    return [[{'id': result['id'], 'texts': result['texts']} for result in sentence] for sentence in mgenre_output]

def get_top_qids(mgenre_output: list[list[dict]]) -> list[str]:
    return [sentence[0]["id"] for sentence in mgenre_output]

def get_length_of_generator(generator: Generator) -> int:
    return sum(1 for value in generator)

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
    total_number_of_examples = 0
    number_of_correctly_predicted_examples = 0
    with open(args.output_path, "w") as file:
        index = 0
        data_iterator=batch_generator_for_iterables(
            batch_size=args.batch_size,
            data=extract_single_mention_mgenre_format(
                extract_single_data_line(args.dataset_path), share_to_return = 0.001
            )
        )
        logging.info('Getting generator length')
        length_of_iterator = get_length_of_generator(data_iterator)
        logging.info('Running evaluation')
        data_iterator=batch_generator_for_iterables(
            batch_size=args.batch_size,
            data=extract_single_mention_mgenre_format(
                extract_single_data_line(args.dataset_path), share_to_return = 0.001
            )
        )
        for (
            mention_ids,
            input_sentences,
            gold_output_qids,
        ) in tqdm(data_iterator, total = length_of_iterator):
            output = inference_with_mgenre(
                sentences=input_sentences,
                model=model,
                knowledge_base=knowledge_base,
                candidate_trie=candidate_trie,
            )
            if index >= 3:
                break
            for mention_id, input_sentence, output_result, gold_output in zip(
                mention_ids, input_sentences, get_json_serialisable_mgenre_output(output), gold_output_qids
            ):
                is_correct = False
                total_number_of_examples += 1
                if output_result[0]['id'] == gold_output:
                    is_correct = True
                    number_of_correctly_predicted_examples += 1
                json_to_write = {
                    "mention_id": mention_id,
                    "input_sentence": input_sentence,
                    "output_result": output_result,
                    "gold_output": gold_output,
                    "is_correct": is_correct
                }
                file.write(json.dumps(json_to_write) + "\n")
            index += 1
    print(f"({number_of_correctly_predicted_examples=}) / ({total_number_of_examples=}) = {number_of_correctly_predicted_examples/total_number_of_examples}")
