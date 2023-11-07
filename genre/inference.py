#!$CONDA_PREFIX/bin/python3

####################
# Required modules #
####################

# Generic
import argparse
import json
import logging
import pathlib
import random

# Libs
from tqdm import tqdm

# Custom
from assessment.entity_linking import evaluate_entity_linking
from config import (
    ENTITY_LINKING_DATASET_PATH,
    MBERT_OUTPUT_PATH,
    MGENRE_KNOWLEDGE_BASE_PATH,
    MGENRE_MARISA_TRIE_PATH,
    MGENRE_MODEL_PATH,
    SPACY_TEST_SET_PATH,
)
from damuel.process_entity_linking_data import (
    extract_single_mention_mgenre_format,
    extract_single_mention_mgenre_format_from_spacy_dataset,
    get_mbert_output_from_spacy_dataset,
    get_iterator_over_texts_from_mbert_output,
)
from damuel.utils import (
    batch_generator_for_iterables,
    load_from_pickle,
)
from GENRE.genre.fairseq_model import mGENRE, mGENREHubInterface
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

    Raises:
        KeyError: If model.sample() raises KeyError.
            This has happened in our experience when the prefix trie contains <unk> tokens which are not found in the knowledge base,
            which happens when the model's tokeniser is unable to encode some page titles.
            The prefix trie creation code has since been updated to prevent this. However, the error handling code has been left in here.
    """
    # See examples_mgenre/examples.ipynb for details on how to use mGENRE
    try:
        return model.sample(
            sentences,
            prefix_allowed_tokens_fn=lambda batch_id, sent: [
                e
                for e in candidate_trie.get(sent.tolist())
                if e < len(model.task.target_dictionary)
            ],
            text_to_id=lambda x: max(
                knowledge_base[tuple(reversed(x.split(" >> ")))],
                key=lambda y: int(y[1:]),
            ),
            marginalize=True,
        )
    except KeyError as e:
        logging.warning("KeyError, isolating first sentence with error.")
        for sentence in sentences:
            try:
                model.sample(
                    [sentence],
                    prefix_allowed_tokens_fn=lambda batch_id, sent: [
                        e
                        for e in candidate_trie.get(sent.tolist())
                        if e < len(model.task.target_dictionary)
                    ],
                    text_to_id=lambda x: max(
                        knowledge_base[tuple(reversed(x.split(" >> ")))],
                        key=lambda y: int(y[1:]),
                    ),
                    marginalize=True,
                )
            except KeyError:
                logging.info(f"Sentence with error: {sentence}")
                sample = model.sample(
                    [sentence],
                    prefix_allowed_tokens_fn=lambda batch_id, sent: [
                        e
                        for e in candidate_trie.get(sent.tolist())
                        if e < len(model.task.target_dictionary)
                    ],
                )
                print(f"model output for sentence: {sample}")
                break
        raise e


def get_json_serialisable_mgenre_output(
    mgenre_output: list[list[dict]],
) -> list[list[dict]]:
    """Returns a JSON serialisable version of the mGENRE output i.e. dropping score tensors.

    Args:
        mgenre_output (list[list[dict]]): mGENRE output from model.sample() with text_to_id and marginalize set to True.

    Returns:
        list[list[dict]]: JSON serialisable version of mGENRE output.
            Format: [[{ 'id': QID,
                        'texts': [f'{page title} >> {language}'],
                      },
                      {option 2},...
                     ], Results for sentence 2, ...
                    ]
    """
    return [
        [{"id": result["id"], "texts": result["texts"]} for result in sentence]
        for sentence in mgenre_output
    ]


def get_top_qids(mgenre_output: list[list[dict]]) -> list[str]:
    """Returns the top QIDs for each sentence in the mGENRE output.

    Args:
        mgenre_output (list[list[dict]]): mGENRE output from model.sample() with text_to_id and marginalize set to True.

    Returns:
        list[str]: List of top QIDs for each sentence.
    """
    return [sentence[0]["id"] for sentence in mgenre_output]


def collate_qids_by_sentence(qids: list[str], input_mbert_dataset: pathlib.Path) -> list[list[dict[str, str | int]]]:
    """Takes a list of the top qid for every mention in the mBERT output, and breaks it into per-sentence lists of Prodigy-formatted mentions. 
    This can be used with evaluate_entity_linking().
    Does this by taking the number of mentions per sentence and the position of each mention from the mBERT dataset.
    
    Args:
        qids (list[str]): List of the top qid for each mention in the mBERT output
        input_mbert_dataset (pathlib.Path): Path to input mBERT dataset
        
    Output:
        list[list[dict[str, str | int]]]: Prodigy-style span labels for each sentence.
            See https://pypi.org/project/nervaluate/ for more details on the Prodigy-style format required.
    
    """
    output_prodigy_spans = []
    qid_index = 0
    with open(input_mbert_dataset, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            sentence_spans = []
            for mention in json_obj["mentions"]:
                sentence_spans.append(
                    {
                        "label": qids[qid_index],
                        "start": mention["start"],
                        "end": mention["end"],
                    }
                )
                qid_index += 1
            output_prodigy_spans.append(sentence_spans)
    return output_prodigy_spans


def get_gold_qids_mbert_dataset(dataset_path: pathlib.Path) -> list[list[dict[str, str | int]]]:
    """Extracts the gold QIDs from the DaMuEL database (in the format stored in the mBERT output dataset) as a Prodigy-style list.
    This can be used as the reference labels in evaluate_entity_linking().
    
    Args:
        dataset_path (pathlib.Path): Path to mBERT output dataset. 
    
    Returns:
        list[list[dict[str, str | int]]]: Prodigy-style gold span labels for each sentence.
            See https://pypi.org/project/nervaluate/ for more details on the Prodigy-style format required.
    """ 
    gold_qids = []
    with open(dataset_path, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            sentence_qids = []
            for mention in json_obj["gold_mentions"]:
                sentence_qids.append(
                    {
                        "label": mention["qid"],
                        "start": mention["start_character"],
                        "end": mention["end_character"],
                    }
                )
            gold_qids.append(sentence_qids)
    return gold_qids


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
        description="Runs DaMuEL dataset through mBERT and mGENRE. Saves the output to a file. Calculates top-1 accuracy."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to DaMuEL dataset",
    )
    parser.add_argument(
        "--spacy_format_dataset",
        action="store_true",
        help="Whether the dataset is in spacy format",
    )
    parser.add_argument(
        "--use_mbert",
        action="store_true",
        help="Whether to do mention detection using mBERT or use the DaMuEL dataset directly with mGENRE",
    )
    parser.add_argument(
        "--do_mbert",
        action="store_true",
        help="Whether to do the mBERT entity detection",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to mGENRE model",
    )
    parser.add_argument(
        "--knowledge_base_path",
        type=str,
        help="Path to knowledge base",
    )
    parser.add_argument(
        "--candidate_marisa_trie_path",
        type=str,
        help="Path to candidate entities Marisa trie",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size",
    )
    parser.add_argument("--logfile_path", type = str, required=True)
    full_el_args = parser.add_argument_group("Full entity linking dataset arguments")
    full_el_args.add_argument(
        "--min_sentence_length",
        help="Minimum sentence length for data extracted from full entity linking dataset",
        type=int,
    )
    full_el_args.add_argument(
        "--share_to_return",
        help="Share of data to be extracted from full entity linking dataset",
        type=float,
    )
    args = parser.parse_args()
    logfile_path = pathlib.Path(args.logfile_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile_path),
                  logging.StreamHandler()],
    )
    knowledge_base_path = (
        pathlib.Path(args.knowledge_base_path)
        if args.knowledge_base_path
        else MGENRE_KNOWLEDGE_BASE_PATH
    )
    if args.dataset_path:
        dataset_path = pathlib.Path(args.dataset_path)
    elif args.spacy_format_dataset:
        dataset_path = SPACY_TEST_SET_PATH
    else:
        dataset_path = ENTITY_LINKING_DATASET_PATH
    model_path = pathlib.Path(args.model_path) if args.model_path else MGENRE_MODEL_PATH
    candidate_trie_path = (
        pathlib.Path(args.candidate_marisa_trie_path)
        if args.candidate_marisa_trie_path
        else MGENRE_MARISA_TRIE_PATH
    )
    if args.output_path:
        output_path = pathlib.Path(args.output_path)
    else:
        if args.use_mbert:
            output_path = (
                SPACY_TEST_SET_PATH.parent / "mbert_mgenre_output_spacy_dataset.jsonl"
            )
        elif args.spacy_format_dataset:
            output_path = (
                SPACY_TEST_SET_PATH.parent / "mgenre_output_spacy_dataset.jsonl"
            )
        else:
            output_path = (
                ENTITY_LINKING_DATASET_PATH.parent / "mgenre_output_full_dataset.jsonl"
            )
    logging.info("Loading model")
    model = mGENRE.from_pretrained(model_path).eval()
    logging.info("Loading knowledge base")
    knowledge_base = load_from_pickle(knowledge_base_path)
    logging.info("Loading candidate trie")
    candidate_trie = load_from_pickle(candidate_trie_path)
    if args.use_mbert:
        if args.do_mbert:
            logging.info("Running mBERT")
            get_mbert_output_from_spacy_dataset(dataset_path, MBERT_OUTPUT_PATH)
        data_iterator = batch_generator_for_iterables(
            args.batch_size,
            get_iterator_over_texts_from_mbert_output(MBERT_OUTPUT_PATH),
        )
    elif args.spacy_format_dataset:
        data_iterator = batch_generator_for_iterables(
            args.batch_size,
            extract_single_mention_mgenre_format_from_spacy_dataset(
                dataset_path=dataset_path
            ),
        )
    else:
        total_number_of_examples = 0
        number_of_correctly_predicted_examples = 0
        random.seed(36)
        data_iterator = batch_generator_for_iterables(
            args.batch_size,
            extract_single_mention_mgenre_format(
                dataset_path=dataset_path,
                break_after=None,
                min_sentence_length=args.min_sentence_length,
                share_to_return=args.share_to_return,
            ),
        )
    logging.info("Running model on data")
    with open(output_path, "w") as file:
        if args.use_mbert:
            predicted_qids = []
            for input_list in tqdm(data_iterator):
                output = inference_with_mgenre(
                    sentences=input_list[0],
                    model=model,
                    knowledge_base=knowledge_base,
                    candidate_trie=candidate_trie,
                )
                predicted_qids += get_top_qids(output)
            logging.info('Collating results')
            predicted_qids_by_sentence = collate_qids_by_sentence(
                predicted_qids, MBERT_OUTPUT_PATH
            )
            gold_qids_by_sentence = get_gold_qids_mbert_dataset(MBERT_OUTPUT_PATH)
            set_of_qids = set(
                (
                    mention["label"]
                    for sentence in gold_qids_by_sentence
                    for mention in sentence
                )
            ) | set(
                (
                    mention["label"]
                    for sentence in predicted_qids_by_sentence
                    for mention in sentence
                )
            )
            with MBERT_OUTPUT_PATH.open('r') as mbertfile:
                for i, line in enumerate(mbertfile): 
                    text_id = json.loads(line)['id']
                    predicted_qids = predicted_qids_by_sentence[i]
                    gold_qids = gold_qids_by_sentence[i]
                    json_to_write = {'text_id': text_id, "predicted_qids": predicted_qids, "gold_qids": gold_qids}
                    file.write(json.dumps(json_to_write) + "\n")
            logging.info('Calculating scores')
            end_to_end_results, chunking_results, linking_results = evaluate_entity_linking(
                labels=gold_qids_by_sentence,
                predictions=predicted_qids_by_sentence,
                set_of_qids=set_of_qids,
            )
            score = {'end_to_end': end_to_end_results, 'chunking': chunking_results, 'linking': linking_results}
            file.write(json.dumps(score))
            logging.info(score)
        else:
            for (
                mention_ids,
                input_sentences,
                gold_output_qids,
            ) in tqdm(data_iterator):
                output = inference_with_mgenre(
                    sentences=input_sentences,
                    model=model,
                    knowledge_base=knowledge_base,
                    candidate_trie=candidate_trie,
                )
                # Compare the top QID with the gold QID
                for mention_id, input_sentence, output_result, gold_output in zip(
                    mention_ids,
                    input_sentences,
                    get_json_serialisable_mgenre_output(output),
                    gold_output_qids,
                ):
                    total_number_of_examples += 1
                    if output_result[0]["id"] == gold_output:
                        is_correct = True
                        number_of_correctly_predicted_examples += 1
                    else:
                        is_correct = False
                    json_to_write = {
                        "mention_id": mention_id,
                        "input_sentence": input_sentence,
                        "output_result": output_result,
                        "gold_output": gold_output,
                        "is_correct": is_correct,
                    }
                    file.write(json.dumps(json_to_write) + "\n")
            logging.info(
                f"({number_of_correctly_predicted_examples=}) / ({total_number_of_examples=}) = {number_of_correctly_predicted_examples/total_number_of_examples}"
            )
