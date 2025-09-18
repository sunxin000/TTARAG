# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import bz2
import json
import os
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from dataset.dataset_adapter import get_dataset_adapter


from loguru import logger
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast
import requests
from prompts.templates import INSTRUCTIONS, IN_CONTEXT_EXAMPLES
from models.config import config
# 从原始文件导入所需函数
from local_evaluation import (
    load_json_file,
    get_system_message,
    attempt_api_call,
    log_response,
    parse_response,
    trim_predictions_to_max_token_length,
    generate_predictions,
    load_data_in_batches,
)

# from CRAG.config import config

tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")

def save_predictions(queries, ground_truths, predictions, evaluation_results=None, save_path=None):
    """保存预测结果到文件，包括评估结果"""
    data = {
        "queries": queries,
        "ground_truths": ground_truths,
        "predictions": predictions,
    }
    
    # 如果有评估结果，添加到数据中
    if evaluation_results:
        data["evaluation_details"] = evaluation_results.get("sample_results", [])
        data["evaluation_summary"] = {
            k: v for k, v in evaluation_results.items() 
            if k != "sample_results"  # 排除样本级别的结果
        }
    
    # save as json file or pickle file
    if save_path.endswith(".json"):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    else:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    logger.info(f"Predictions and evaluation results saved to {save_path}")

def load_predictions(save_path):
    """从文件加载预测结果"""
    # load pickle file or load json file
    if save_path.endswith(".json"):
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
    logger.info(f"Predictions loaded from {save_path}")
    return data["queries"], data["ground_truths"], data["predictions"]

def evaluate_single_prediction(args):
    """评估单个预测的函数，用于并行处理"""
    idx, query, ground_truths, prediction, evaluation_model_name = args
    
    if isinstance(ground_truths, (str, int, float)):
        ground_truths = [str(ground_truths)]
    elif isinstance(ground_truths, list):
        ground_truths = [str(gt) for gt in ground_truths]
        
    prediction = trim_predictions_to_max_token_length(prediction)
    prediction = prediction.strip()
    prediction_lowercase = prediction.lower()
    
    result = {
        "idx": idx,
        "correct": False,
        "miss": False,
        "api_call": False,
        "exact_match": False,
        "invalid_match": False,
        "i_dont_know": False
    }
    
    if "i don't know" in prediction_lowercase:
        result["miss"] = True
        result["i_dont_know"] = True
        return result
        
    for ground_truth in ground_truths:
        ground_truth_lowercase = ground_truth.lower()
        
        # 完全匹配检查
        if prediction_lowercase == ground_truth_lowercase:
            result["correct"] = True
            result["exact_match"] = True
            return result
            
        # Invalid 匹配检查
        elif "invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase:
            result["correct"] = True
            result["invalid_match"] = True
            return result
            
        # Invalid 不匹配检查
        elif ("invalid" in prediction_lowercase) != ("invalid" in ground_truth_lowercase):
            continue
            
        # 需要 API 调用
        else:
            result["api_call"] = True
            messages = [
                {"role": "system", "content": get_system_message()},
                {
                    "role": "user",
                    "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
                },
            ]
            
            response = attempt_api_call(evaluation_model_name, messages)
            if response:
                log_response(messages, response)
                explanation, accuracy = parse_response(response)
                if accuracy == 1:
                    result["correct"] = True
                    return result
                    
    return result

def evaluate_predictions(queries, ground_truths_list, predictions, evaluation_model_name, 
                        max_workers=10, domains=None, existing_evaluation=None):
    n = len(predictions)
    
    # 初始化整体统计
    stats = {
        "total_samples": n,
        "api_calls": 0,
        "exact_matches": 0,
        "invalid_matches": 0,
        "i_dont_know_count": 0,
        "api_match_count": 0
    }
    
    # 初始化按domain统计的结果
    domain_stats = {}
    if domains:
        unique_domains = set(domains)
        for domain in unique_domains:
            domain_stats[domain] = {
                "total_samples": 0,
                "n_correct": 0,
                "n_miss": 0,
                "n_hallucination": 0,
                "api_calls": 0,
                "exact_matches": 0,
                "invalid_matches": 0,
                "i_dont_know_count": 0,
                "api_match_count": 0
            }
    
    logger.info(f"Starting parallel evaluation of {n} predictions with {max_workers} workers...")
    
    # 使用列表存储所有结果
    all_results = [None] * n
    
    if existing_evaluation and "evaluation_details" in existing_evaluation:
    # if False:
        logger.info("Using existing evaluation results...")
        # 使用现有的评估结果
        all_results = existing_evaluation["evaluation_details"]
        # 添加domain信息到现有结果中
        for idx, result in enumerate(all_results):
            result["domain"] = domains[idx] if domains else "unknown"
            
            # 更新整体统计信息
            if result["api_call"]:
                stats["api_calls"] += 1
            if result["exact_match"]:
                stats["exact_matches"] += 1
            if result["invalid_match"]:
                stats["invalid_matches"] += 1
            if result["i_dont_know"]:
                stats["i_dont_know_count"] += 1
            if result["correct"] and result["api_call"]:
                stats["api_match_count"] += 1
                
            # 更新domain统计信息
            if domains:
                domain = domains[idx]
                domain_stats[domain]["total_samples"] += 1
                if result["api_call"]:
                    domain_stats[domain]["api_calls"] += 1
                if result["exact_match"]:
                    domain_stats[domain]["exact_matches"] += 1
                if result["invalid_match"]:
                    domain_stats[domain]["invalid_matches"] += 1
                if result["i_dont_know"]:
                    domain_stats[domain]["i_dont_know_count"] += 1
                if result["correct"] and result["api_call"]:
                    domain_stats[domain]["api_match_count"] += 1
                if result["correct"]:
                    domain_stats[domain]["n_correct"] += 1
                if result["miss"]:
                    domain_stats[domain]["n_miss"] += 1
                if not result["correct"] and not result["miss"]:
                    domain_stats[domain]["n_hallucination"] += 1
    else:
        logger.info("No existing evaluation results found, performing new evaluation...")
        # 执行原有的评估逻辑
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    evaluate_single_prediction, 
                    (i, queries[i], ground_truths_list[i], predictions[i], evaluation_model_name)
                ): i for i in range(n)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=n, desc="Evaluating Predictions"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    result["domain"] = domains[idx] if domains else "unknown"
                    all_results[idx] = result
                    
                    # 更新整体统计信息
                    if result["api_call"]:
                        stats["api_calls"] += 1
                    if result["exact_match"]:
                        stats["exact_matches"] += 1
                    if result["invalid_match"]:
                        stats["invalid_matches"] += 1
                    if result["i_dont_know"]:
                        stats["i_dont_know_count"] += 1
                    if result["correct"] and result["api_call"]:
                        stats["api_match_count"] += 1
                        
                    # 更新domain统计信息
                    if domains:
                        domain = domains[idx]
                        domain_stats[domain]["total_samples"] += 1
                        if result["api_call"]:
                            domain_stats[domain]["api_calls"] += 1
                        if result["exact_match"]:
                            domain_stats[domain]["exact_matches"] += 1
                        if result["invalid_match"]:
                            domain_stats[domain]["invalid_matches"] += 1
                        if result["i_dont_know"]:
                            domain_stats[domain]["i_dont_know_count"] += 1
                        if result["correct"] and result["api_call"]:
                            domain_stats[domain]["api_match_count"] += 1
                        if result["correct"]:
                            domain_stats[domain]["n_correct"] += 1
                        if result["miss"]:
                            domain_stats[domain]["n_miss"] += 1
                        if not result["correct"] and not result["miss"]:
                            domain_stats[domain]["n_hallucination"] += 1
                            
                except Exception as e:
                    logger.error(f"Error processing prediction {idx}: {str(e)}")
                    all_results[idx] = {
                        "idx": idx,
                        "correct": False,
                        "miss": True,
                        "error": str(e),
                        "domain": domains[idx] if domains else "unknown"
                    }
    
    # 计算整体结果
    n_correct = sum(1 for r in all_results if r["correct"])
    n_miss = sum(1 for r in all_results if r["miss"])
    
    final_results = {
        "overall": {
            "score": (2 * n_correct + n_miss) / n - 1,
            "accuracy": n_correct / n,
            "hallucination": (n - n_correct - n_miss) / n,
            "missing": n_miss / n,
            "n_miss": n_miss,
            "n_correct": n_correct,
            "n_hallucination": n - n_correct - n_miss,
            "total": n,
            **stats
        },
        "sample_results": all_results
    }
    
    # 添加每个domain的结果
    if domains:
        final_results["by_domain"] = {}
        for domain, stats in domain_stats.items():
            if stats["total_samples"] > 0:  # 只输出有样本的domain
                final_results["by_domain"][domain] = {
                    "score": (2 * stats["n_correct"] + stats["n_miss"]) / stats["total_samples"] - 1,
                    "accuracy": stats["n_correct"] / stats["total_samples"],
                    "hallucination": stats["n_hallucination"] / stats["total_samples"],
                    "missing": stats["n_miss"] / stats["total_samples"],
                    **stats
                }
    
    # 输出详细统计信息
    logger.info("\n" + "="*50)
    logger.info("Overall Evaluation Statistics:")
    logger.info(f"Total samples processed: {stats['total_samples']}")
    logger.info(f"Exact matches found: {stats['exact_matches']}")
    logger.info(f"Invalid matches found: {stats['invalid_matches']}")
    logger.info(f"I don't know responses: {stats['i_dont_know_count']}")
    logger.info(f"Total API calls made: {stats['api_calls']}")
    logger.info(f"Successful API matches: {stats['api_match_count']}")
    
    if domains:
        logger.info("\nResults by Domain:")
        for domain, stats in final_results["by_domain"].items():
            logger.info(f"\n{domain} Statistics:")
            logger.info(f"Total samples: {stats['total_samples']}")
            logger.info(f"Accuracy: {stats['accuracy']:.3f}")
            logger.info(f"Score: {stats['score']:.3f}")
            logger.info(f"Hallucination rate: {stats['hallucination']:.3f}")
            logger.info(f"Missing rate: {stats['missing']:.3f}")
    
    return final_results

def generate_predictions_path(config, args):
    """根据配置生成预测结果文件路径"""
    components = []
    
    # 添加数据集名称
    components.append(config.dataset.name)

        # 添加模型类型标识
    from models.user_config import UserModel
    model_type = ""
    if UserModel.__name__ == "CoTModel":
        model_type = "cot"
    elif UserModel.__name__ == "ICLModel":
        model_type = "icl"
    elif UserModel.__name__ == "RAGModel":
        model_type = "rag"
    components.append(model_type)

    # 添加模型名称（取最后一部分）
    model_name = config.rag.model_name.split('/')[-1].lower()
    components.append(model_name)
    
    # 添加PEFT相关信息
    if config.tta.use_peft:
        components.append("peft")
        if config.tta.peft_method == "lora":
            components.append(f"lora_r{config.tta.lora_r}")
            components.append(f"lora_alpha{config.tta.lora_alpha}")
            components.append(f"dropout{config.tta.lora_dropout}")
        elif config.tta.peft_method == "prefix":
            components.append(f"prefix_tokens{config.tta.num_virtual_tokens}")
            if config.tta.token_dim:
                components.append(f"token_dim{config.tta.token_dim}")
            if config.tta.num_transformer_submodules:
                components.append(f"trans_submod{config.tta.num_transformer_submodules}")
    
    # 添加sentence selection相关信息
    if config.tta.use_sentence_selection:
        components.append(f"select_{config.tta.selection_method}")
        if config.tta.selection_method == "rerank":
            components.append(config.tta.reranker_type)
            if config.tta.reranker_type == "cross_encoder":
                reranker_name = config.tta.cross_encoder_name.split('/')[-1].lower()
                components.append(reranker_name)
        elif config.tta.selection_method == "summarize":
            summarizer_name = config.tta.summarizer_name.split('/')[-1].lower()
            components.append(summarizer_name)
        components.append(f"top{config.tta.max_selected_sentences}")
    
    # 如果使用TTA，添加TTA相关配置
    if args.do_tta:
        components.append("tta")
        components.append(f"lr{config.tta.learning_rate}")
        components.append(f"pairs{config.tta.max_adapt_pairs}")
        components.append(f"accum{config.tta.accumulation_steps}")
        
        if config.tta.use_dexperts:
            components.append("dexperts")
            expert_name = config.tta.expert_model_name.split('/')[-1].lower()
            components.append(f"exp_{expert_name}")
            components.append(f"alpha{config.tta.alpha}")
    
    # 添加时间戳
    timestamp = datetime.now().strftime("%m%d_%H%M")
    components.append(timestamp)
    
    # 组合文件名，使用下划线连接所有组件
    filename = '_'.join(components) + '.json'
    
    # 确保predictions目录存在
    os.makedirs("predictions", exist_ok=True)
    
    return os.path.join("predictions", filename)

def save_evaluation_results(evaluation_results, config, args, predictions_path):
    """保存评估结果和配置参数"""
    # 创建完整的结果字典
    full_results = {
        "evaluation_results": evaluation_results,
        "config": {
            "rag": config.rag.dict(),
            "tta": config.tta.dict(),
            "dataset": config.dataset.dict(),
            "parallel_eval": config.parallel_eval.dict()
        },
        "command_line_args": vars(args),
        "predictions_path": predictions_path,
        "timestamp": datetime.now().isoformat()
    }
    
    # 创建results目录
    os.makedirs("results", exist_ok=True)
    dataset_name = config.dataset.name
    
    # 使用与predictions相同的基础名称
    base_name = os.path.splitext(os.path.basename(predictions_path))[0]
    results_path = os.path.join(f"results/{dataset_name}", f"{base_name}_eval.json")
    
    # 保存结果
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=4)
    
    logger.info(f"Evaluation results and config saved to {results_path}")
    return results_path

if __name__ == "__main__":
    from models.user_config import UserModel
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_generate", action="store_true", help="Whether to generate new predictions")
    parser.add_argument("--predictions_path", type=str, help="Path to save/load predictions")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--do_tta", action="store_true", help="Whether to use TTA")
    
    # RAG 相关参数
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--batch_size", type=int, help="Submission batch size")
    parser.add_argument("--tensor_parallel_size", type=int, help="VLLM tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, help="GPU memory utilization")
    
    # TTA 相关参数
    parser.add_argument("--learning_rate", type=float, help="TTA learning rate")
    parser.add_argument("--max_adapt_pairs", type=int, help="Maximum adaptation pairs")
    parser.add_argument("--use_dexperts", action="store_true", help="Whether to use DExperts")
    parser.add_argument("--expert_model_name", type=str, help="Expert model name for DExperts")
    parser.add_argument("--alpha", type=float, help="Alpha value for DExperts")
    parser.add_argument("--reranker_type", type=str, help="Reranker type for sentence selection")


    parser.add_argument("--max_selected_sentences", type=int, help="Maximum selected sentences")
    parser.add_argument("--cross_encoder_name", type=str, help="Cross encoder name for sentence selection")
    parser.add_argument("--summarizer_name", type=str, help="Summarizer name for sentence selection")
    parser.add_argument("--max_summary_length", type=int, help="Maximum summary length")
    parser.add_argument("--min_summary_length", type=int, help="Minimum summary length")
    parser.add_argument("--selection_method", type=str, help="Selection method")
    parser.add_argument("--use_sentence_selection", action="store_true", help="Whether to use sentence selection")

    
    parser.add_argument("--peft_method", type=str, help="PEFT method")
    parser.add_argument("--lora_r", type=int, help="Lora r")
    parser.add_argument("--lora_alpha", type=int, help="Lora alpha")
    parser.add_argument("--lora_dropout", type=float, help="Lora dropout")
    parser.add_argument("--use_peft", action="store_true", help="Whether to use PEFT")
    # 数据集相关参数
    parser.add_argument("--dataset", type=str, choices=["crag", "pubmedqa", "medrag"], help="Dataset name")
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    
    # 在现有的TTA相关参数中添加prefix tuning的参数
    parser.add_argument("--num_virtual_tokens", type=int, help="Number of virtual tokens for prefix tuning")
    parser.add_argument("--token_dim", type=int, help="Token dimension for prefix tuning")
    parser.add_argument("--num_transformer_submodules", type=int, help="Number of transformer submodules")
    parser.add_argument("--num_attention_heads", type=int, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, help="Number of layers")
    parser.add_argument("--encoder_hidden_size", type=int, help="Encoder hidden size")
    parser.add_argument("--prefix_projection", type=bool, help="Whether to use prefix projection")
    

    # 添加P-tuning特定参数
    parser.add_argument("--prompt_encoder_type", type=str, choices=["lstm", "mlp"], help="Type of prompt encoder for P-tuning")
    parser.add_argument("--prompt_num_layers", type=int, help="Number of layers in prompt encoder for P-tuning")
    parser.add_argument("--prompt_dropout", type=float, help="Dropout rate in prompt encoder for P-tuning")
    parser.add_argument("--prompt_hidden_size", type=int, help="Hidden size of prompt encoder for P-tuning")
    
    parser.add_argument("--wosegment", action="store_true", help="Whether to skip setting prefix labels to -100")
    
    args = parser.parse_args()
    
    # 更新配置
    from models.config import config, update_config_from_args
    config = update_config_from_args(config, args)
    
    # 如果未指定predictions_path，则自动生成
    if args.predictions_path is None:
        args.predictions_path = generate_predictions_path(config, args)
    
    logger.info(f"Using predictions path: {args.predictions_path}")
    
    if config.dataset.name == "crag":
        DATASET_PATH = "/ossfs/workspace/CRAG/crag_task_1_and_2_dev_v4.jsonl.bz2"
    else:
        DATASET_PATH = None
    EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "Qwen25_72B_Instruct_awq_vllm_l20")

    if args.do_generate:
        # Generate predictions
        participant_model = UserModel(do_tta=args.do_tta)
        queries, ground_truths, predictions = generate_predictions(DATASET_PATH, participant_model)
        # 先保存原始预测结果
        save_predictions(queries, ground_truths, predictions, save_path=args.predictions_path)
        existing_evaluation = None
    else:
        # Load existing predictions
        with open(args.predictions_path, 'r') as f:
            data = json.load(f)
        queries = data["queries"]
        ground_truths = data["ground_truths"]
        predictions = data["predictions"]
        # 获取已有的评估结果（如果存在）
        existing_evaluation = {
            "evaluation_details": data.get("evaluation_details", []),
            "evaluation_summary": data.get("evaluation_summary", {})
        } if "evaluation_details" in data else None
    
        if existing_evaluation is None:
            # 构建可能的results文件路径
            base_name = os.path.splitext(os.path.basename(args.predictions_path))[0]
            results_path = os.path.join(f"results/{config.dataset.name}", f"{base_name}_eval.json")
            
            if os.path.exists(results_path):
                logger.info(f"Found results file at {results_path}")
                try:
                    with open(results_path, 'r') as f:
                        results_data = json.load(f)
                        if "evaluation_results" in results_data:
                            # existing_evaluation = results_data["evaluation_results"]["sample_results"]
                            logger.info(f"Loaded existing evaluation from {results_path}")
                            existing_evaluation = {"evaluation_details": results_data["evaluation_results"]["sample_results"]}
                except Exception as e:
                    logger.warning(f"Failed to load results from {results_path}: {e}")
            else:
                logger.info(f"No existing evaluation found in predictions file {results_path}")
    # 获取domains信息（如果数据集适配器提供了这个信息）
    # try:
    dataset_adapter = get_dataset_adapter(config.dataset.name)
    data = dataset_adapter.load_data()
    domains = [dataset_adapter.format_sample(sample)["domain"] for sample in data]
    # print(domains)
    # except:
    #     domains = None
    #     logger.warning("Could not get domain information from dataset adapter")
    
    # Evaluate Predictions
    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, EVALUATION_MODEL_NAME, 
        max_workers=args.max_workers,
        domains=domains,
        existing_evaluation=existing_evaluation
    )

    # 更新预测文件，加入评估结果
    save_predictions(
        queries, 
        ground_truths, 
        predictions, 
        evaluation_results=evaluation_results,
        save_path=args.predictions_path
    )

    # 保存评估结果和配置到单独的文件
    results_path = save_evaluation_results(
        evaluation_results,
        config,
        args,
        args.predictions_path
    )