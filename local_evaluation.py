# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import bz2
import json
import os
import re
from datetime import datetime

from loguru import logger
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast
import requests
from prompts.templates import INSTRUCTIONS, IN_CONTEXT_EXAMPLES
from dataset.dataset_adapter import get_dataset_adapter
from models.config import config

tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    from models.user_config import UserModel

    if UserModel.__name__ == "CoTModel":
        # return COT_INSTRUCTIONS + "\n" + COT_IN_CONTEXT_EXAMPLES
        return COT_INSTRUCTIONS + "\n" + COT_IN_CONTEXT_EXAMPLES
    else:
        return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES


def attempt_api_call(model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    url = "xxxxxx"

    for attempt in range(max_retries):
        try:
            data = {"messages": messages, "temperature": 0.0}

            body = {
                "serviceCode": model_name,
                "uri": "v1",
                "attributes": {
                    "_TIMEOUT_": "50000",
                    "_ROUTE_": "MAYA",
                    "_TOKEN_": "111",
                },
                "params": {"features": {"query": json.dumps(data)}},
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, data=json.dumps(body))

            if response.status_code == 200:
                result = response.json()
                return json.loads(result["resultMap"]["attributes"]["result"])[
                    "choices"
                ][0]["message"]["content"]
            else:
                logger.warning(f"API call failed with status {response.status_code}")

        except Exception as e:
            logger.warning(
                f"API call failed on attempt {attempt + 1}: {str(e)}, retrying..."
            )

    return None


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    # file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    # file_path = os.path.join(output_directory, file_name)
    # with open(file_path, "w") as f:
    #     json.dump({"messages": messages, "response": response}, f)


def parse_response(response: str):
    """Return a tuple of (explanation, score) from the response."""
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to a reasonable length while preserving complete sentences"""
    max_token_length = 512  # 增加到更合理的长度

    tokenized_prediction = tokenizer.encode(prediction)
    if len(tokenized_prediction) <= max_token_length:
        return prediction

    # 截取前max_token_length个token
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)

    # 尝试在句号处截断，确保句子完整性
    last_period = trimmed_prediction.rfind(".")
    if last_period > 0:
        trimmed_prediction = trimmed_prediction[: last_period + 1]

    return trimmed_prediction


def evaluate_predictions(
    queries, ground_truths_list, predictions, evaluation_model_name
):
    """Evaluates predictions using Qwen model."""
    n_miss, n_correct = 0, 0
    system_message = get_system_message()

    # 添加统计信息
    stats = {
        "total_samples": len(predictions),
        "api_calls": 0,
        "exact_matches": 0,
        "invalid_matches": 0,
        "i_dont_know_count": 0,
        "api_match_count": 0,
    }

    logger.info(f"Starting evaluation of {stats['total_samples']} predictions...")

    for _idx, prediction in enumerate(tqdm(predictions, desc="Evaluating Predictions")):
        query = queries[_idx]
        ground_truths = ground_truths_list[_idx]
        if isinstance(ground_truths, (str, int, float)):
            ground_truths = [str(ground_truths)]  # 转换为字符串列表
        elif isinstance(ground_truths, list):
            ground_truths = [str(gt) for gt in ground_truths]
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Sample {_idx + 1}/{stats['total_samples']}:")
        logger.info(f"Query: {query}")
        logger.info(f"Ground truths: {ground_truths}")

        prediction = trim_predictions_to_max_token_length(prediction)
        prediction = prediction.strip()
        logger.info(f"Prediction: {prediction}")

        prediction_lowercase = prediction.lower()

        if "i don't know" in prediction_lowercase:
            logger.info("Result: I don't know response detected")
            stats["i_dont_know_count"] += 1
            n_miss += 1
            continue

        accuracy = -1
        for ground_truth in ground_truths:
            ground_truth_lowercase = ground_truth.lower()

            logger.info(f"\nComparing with ground truth: {ground_truth}")

            # 完全匹配检查
            if prediction_lowercase == ground_truth_lowercase:
                logger.info("Result: Exact match found!")
                accuracy = 1
                stats["exact_matches"] += 1
                break

            # Invalid 匹配检查
            elif (
                "invalid" in prediction_lowercase
                and "invalid" in ground_truth_lowercase
            ):
                logger.info("Result: Both contain 'invalid' - match found!")
                accuracy = 1
                stats["invalid_matches"] += 1
                break

            # Invalid 不匹配检查
            elif (
                "invalid" in prediction_lowercase
                and "invalid" not in ground_truth_lowercase
            ):
                logger.info(
                    "Result: Prediction contains 'invalid' but ground truth doesn't - no match"
                )
                accuracy = 0
                continue
            elif (
                "invalid" not in prediction_lowercase
                and "invalid" in ground_truth_lowercase
            ):
                logger.info(
                    "Result: Ground truth contains 'invalid' but prediction doesn't - no match"
                )
                accuracy = 0
                continue

            # 需要 API 调用
            else:
                logger.info(
                    "No simple match found, calling API for semantic comparison..."
                )
                messages = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
                    },
                ]
                stats["api_calls"] += 1
                response = attempt_api_call(evaluation_model_name, messages)

                if response:
                    # log_response(messages, response)
                    explanation, accuracy = parse_response(response)
                    logger.info(
                        f"API Response - Score: {accuracy}, Explanation: {explanation}"
                    )
                    if accuracy == 1:
                        stats["api_match_count"] += 1
                        break
                else:
                    logger.warning("API call failed after all retries")

        if accuracy == 1:
            n_correct += 1

    # 计算最终结果
    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": n - n_correct - n_miss,
        "total": n,
    }

    # 输出详细统计信息
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Statistics:")
    logger.info(f"Total samples processed: {stats['total_samples']}")
    logger.info(f"Exact matches found: {stats['exact_matches']}")
    logger.info(f"Invalid matches found: {stats['invalid_matches']}")
    logger.info(f"I don't know responses: {stats['i_dont_know_count']}")
    logger.info(f"Total API calls made: {stats['api_calls']}")
    logger.info(f"Successful API matches: {stats['api_match_count']}")
    logger.info("\nFinal Results:")
    logger.info(results)

    return results


def generate_predictions(dataset_path, participant_model):
    """Generate predictions using the participant model."""
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    # Get appropriate dataset adapter
    adapter = get_dataset_adapter(
        config.dataset.name,
        # dataset_path=dataset_path,
    )

    for batch in tqdm(
        adapter.load_data_in_batches(batch_size), desc="Generating predictions"
    ):
        batch_ground_truths = batch.pop("answer")
        batch_predictions = participant_model.batch_generate_answer(batch)

        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)

    if hasattr(participant_model, "save_retrieval_history"):
        participant_model.save_retrieval_history()

    return queries, ground_truths, predictions


def load_data_in_batches(dataset_path, batch_size):
    """Load data in batches from a compressed file."""

    def initialize_batch():
        return {
            "interaction_id": [],
            "query": [],
            "search_results": [],
            "query_time": [],
            "answer": [],
        }

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


if __name__ == "__main__":
    from models.user_config import UserModel

    DATASET_PATH = "/ossfs/workspace/CRAG/example_data/dev_data.jsonl.bz2"
    EVALUATION_MODEL_NAME = os.getenv(
        "EVALUATION_MODEL_NAME", "Qwen25_72B_Instruct_awq_vllm_l20"
    )

    # Generate predictions
    participant_model = UserModel()
    queries, ground_truths, predictions = generate_predictions(
        DATASET_PATH, participant_model
    )

    # Evaluate Predictions
    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, EVALUATION_MODEL_NAME
    )
