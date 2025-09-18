# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from .tta_module import TTAModule
from .config import config
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from loguru import logger
import os
import json
import time
######################################################################################################
######################################################################################################
###
### Please pay special attention to the comments that start with "TUNE THIS VARIABLE"
###                        as they depend on your model and the available GPU resources.
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---
def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits



class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class RAGModel:
    """
    An example RAGModel
    """
    def __init__(self, do_tta=False):
        self.do_tta = do_tta
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()
        if do_tta:
            self.tta_module = TTAModule(
                self.llm, 
                self.tokenizer,
            )
        self.retrieval_history = []  # 添加存储检索历史的列表

    def initialize_models(self):
        """Initialize the models required for RAG."""
        self.model_name = config.rag.model_name
        
        if self.do_tta:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cuda:0",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            
        else:        
            self.llm = vllm.LLM(
                self.model_name,
                worker_use_ray=False,
                tensor_parallel_size=config.rag.vllm_tensor_parallel_size, 
                gpu_memory_utilization=config.rag.vllm_gpu_memory_utilization, 
                trust_remote_code=True,
                dtype="half", # note: update the dtype based on the available GPU
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()
        
        self.sentence_encoder = SentenceTransformer(
            config.rag.embedding_model_name,
            device="cuda"
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_encoder.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = config.rag.submission_batch_size
        return self.batch_size

    def retrieve_relevant_sentences(self, batch_interaction_ids, queries, batch_search_results, query_times):
        """
        检索与查询相关的句子
        
        Parameters:
            batch_interaction_ids (List[str]): 交互ID列表
            queries (List[str]): 查询列表
            batch_search_results (List[List[Dict]]): 搜索结果批次列表
            query_times (List[str]): 查询时间列表
            
        Returns:
            List[List[str]]: 每个查询对应的相关句子列表
        """
        if config.dataset.name == 'crag':
            # 抽取chunks
            chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
                batch_interaction_ids, batch_search_results
            )

            # 计算chunks的embeddings
            chunk_embeddings = self.calculate_embeddings(chunks)

            # 计算queries的embeddings  
            query_embeddings = self.calculate_embeddings(queries)

            # 为每个查询检索相关句子
            batch_retrieval_results = []
            for _idx, interaction_id in enumerate(batch_interaction_ids):
                query_embedding = query_embeddings[_idx]

                # 找出属于当前interaction_id的chunks
                relevant_chunks_mask = chunk_interaction_ids == interaction_id
                relevant_chunks = chunks[relevant_chunks_mask]
                relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

                # 计算相似度并获取top-N结果
                cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)
                retrieval_results = relevant_chunks[
                    (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
                ]

                # 保存检索结果
                self.retrieval_history.append({
                    'interaction_id': interaction_id,
                    'query': queries[_idx],
                    'retrieved_sentences': retrieval_results.tolist(),  # 转换为普通列表以便序列化
                    'query_time': query_times[_idx]
                })
                
                batch_retrieval_results.append(retrieval_results)
        else:
            batch_retrieval_results = [[y["page_result"] for y in x[:20]] for x in batch_search_results]
        
        return batch_retrieval_results

    def generate_with_dexperts(self, prompt, retrieval_results):
        """
        使用DExperts方法生成答案
        
        Parameters:
            prompt (str): 格式化后的提示词
            retrieval_results (List[str]): 检索到的相关句子
            
        Returns:
            str: 生成的答案
        """
        # 首先适应专家模型
        adapted_model = self.tta_module.adapt_model(retrieval_results)
        if adapted_model is None:
            return "I don't know"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        # 使用DExperts引导生成
        outputs = []
        for i in range(config.rag.generation_params.max_new_tokens):
            # 获取基础模型logits
            with torch.no_grad():
                base_outputs = self.llm(**inputs)
                base_logits = base_outputs.logits[:, -1, :]
            
            # 获取专家和反专家logits
            expert_logits, antiexpert_logits = self.tta_module._get_dexperts_logits(
                inputs.input_ids,
                inputs.attention_mask
            )
            
            # 应用DExperts公式
            next_token_logits = (
                base_logits + 
                self.tta_module.alpha * (expert_logits - antiexpert_logits)
            )
            
            # 采样下一个token
            if config.rag.generation_params.temperature != 1.0:
                next_token_logits = next_token_logits / config.rag.generation_params.temperature
            if config.rag.generation_params.top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(
                    next_token_logits, 
                    top_p=config.rag.generation_params.top_p
                )
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            outputs.append(next_token.item())
            inputs["input_ids"] = torch.cat([inputs.input_ids, next_token], dim=-1)
            inputs["attention_mask"] = torch.cat([
                inputs.attention_mask,
                torch.ones((inputs.attention_mask.shape[0], 1), dtype=torch.long, device=inputs.attention_mask.device)
            ], dim=-1)
        
        return self.tokenizer.decode(outputs, skip_special_tokens=True)

    def generate_with_traditional_tta(self, prompt, retrieval_results):
        """
        使用传统TTA方法生成答案
        
        Parameters:
            prompt (str): 格式化后的提示词
            retrieval_results (List[str]): 检索到的相关句子
            
        Returns:
            str: 生成的答案
        """
        # 首先适应模型
        adapted_model = self.tta_module.adapt_model(retrieval_results)
        if adapted_model is None:
            # 如果适应失败,使用原始模型
            adapted_model = self.llm
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(adapted_model.device)
        outputs = adapted_model.generate(
            **inputs,
            max_new_tokens=config.rag.generation_params.max_new_tokens,
            temperature=config.rag.generation_params.temperature,
            top_p=config.rag.generation_params.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

    def generate_with_tta(self, formatted_prompts, batch_retrieval_results):
        """
        使用TTA(Test Time Adaptation)生成答案
        
        Parameters:
            formatted_prompts (List[str]): 格式化后的提示词列表
            batch_retrieval_results (List[List[str]]): 检索到的相关句子列表
            
        Returns:
            List[str]: 生成的答案列表
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        responses = []
        for prompt, retrieval_results in zip(formatted_prompts, batch_retrieval_results):
            if config.tta.use_dexperts:
                response = self.generate_with_dexperts(prompt, retrieval_results)
            else:
                response = self.generate_with_traditional_tta(prompt, retrieval_results)
            responses.append(response)
        
        return responses

    def generate_with_vllm(self, formatted_prompts):
        """
        使用VLLM生成答案
        
        Parameters:
            formatted_prompts (List[str]): 格式化后的提示词列表
            
        Returns:
            List[str]: 生成的答案列表
        """
        sampling_params = vllm.SamplingParams(
            n=1,
            temperature=config.rag.generation_params.temperature,
            top_p=config.rag.generation_params.top_p,
            max_tokens=config.rag.generation_params.max_new_tokens,
        )
        responses = self.llm.generate(formatted_prompts, sampling_params)
        return [response.outputs[0].text for response in responses]

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        为一批查询生成答案
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # 检索相关句子
        start_time = time.time()
        batch_retrieval_results = self.retrieve_relevant_sentences(
            batch_interaction_ids,
            queries, 
            batch_search_results,
            query_times
        )
        end_time = time.time()
        logger.info(f"Retrieval time: {end_time - start_time} seconds")

        # 准备提示词
        start_time = time.time()
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)
        end_time = time.time()
        logger.info(f"format time: {end_time - start_time} seconds")
        # 生成答案
        start_time = time.time()
        answers = self.generate_with_tta(formatted_prompts, batch_retrieval_results) if self.do_tta \
            else self.generate_with_vllm(formatted_prompts)
        end_time = time.time()
        logger.info(f"Generation time: {end_time - start_time} seconds")
            
        # 将答案添加到检索历史中
        if config.dataset.name=='crag':
            for idx, answer in enumerate(answers):
                self.retrieval_history[-(len(answers)-idx)]['answer'] = answer
            
        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """
        # system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. Explain the reasoning behind your answers."
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. There is no need to explain the reasoning behind your answers."
        # system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        # 定义不同模型族的模板
        MODEL_TEMPLATES = {
            # Llama 系列
            "llama": {
                "base": {  # 基础模型 (Llama-3.1-8B-Instruct)
                    "template": "{system}\n\n {user}\n",
                    "is_chat_model": False
                },
                "instruct": {  # instruct模型 (Llama-3.1-8B-Instruct)
                    "template": "<s>[INST] {system}\n\n{user} [/INST]",
                    "is_chat_model": True
                }
            },
            # Mistral 系列
            "mistral": {
                "base": {  # 基础模型 (Mistral-7B-v0.1)
                    "template": "System: {system}\n\nUser: {user}\nAssistant:",
                    "is_chat_model": False
                },
                "instruct": {  # instruct模型 (Mistral-7B-Instruct-v0.1)
                    "template": "<s>[INST] {system}\n\n{user} [/INST]",
                    "is_chat_model": True
                }
            },
            # Gemma 系列
            "gemma": {
                "base": {  # 基础模型 (gemma-2b, gemma-7b)
                    "template": "{system}\n\nUser: {user}\nModel:",
                    "is_chat_model": False
                },
                "instruct": {  # instruct模型 (gemma-2b-it, gemma-7b-it)
                    "template": "<start_of_turn>user\n{system}\n{user}<end_of_turn>\n<start_of_turn>model",
                    "is_chat_model": True
                }
            },
            # Qwen 系列
            "qwen": {
                "instruct": {  # instruct模型 (qwen2.5-7b-instruct)
                    "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
                    "is_chat_model": True
                }
            },
            # ChatGLM 系列
            "chatglm": {
                "base": {
                    "template": "[gMASK]sop<|system|>\n {system}<|user|>\n {user}<|assistant|>",
                    "is_chat_model": True
                }
            },
            # Falcon 系列
            "falcon": {
                "base": {  # 基础模型 (falcon-7b)
                    "template": "System: {system}\n\nUser: {user}\nAssistant:",
                    "is_chat_model": False
                },
                "instruct": {  # instruct模型 (falcon-7b-instruct)
                    "template": "System: {system}\n\nHuman: {user}\nAssistant:",
                    "is_chat_model": True
                }
            }
        }

        def get_model_template(model_name):
            """根据模型名称获取对应的模板"""
            model_name = model_name.lower()
            
            # 确定模型族
            if "llama" in model_name:
                family = "llama"
            elif "mistral" in model_name:
                family = "mistral"
            elif "gemma" in model_name:
                family = "gemma"
            elif "qwen" in model_name:
                family = "qwen"
            elif "chatglm" in model_name:
                family = "chatglm"
            elif "falcon" in model_name:
                family = "falcon"
            else:
                return MODEL_TEMPLATES["llama"]["base"]  # 默认使用 llama instruct 格式
            
            # 确定是否是 instruct 版本
            if any(x in model_name for x in ["-it", "instruct", "-i-"]):
                variant = "instruct"
            else:
                variant = "base"
                
            # 获取模板，如果没有对应变体则返回族的第一个可用模板
            family_templates = MODEL_TEMPLATES[family]
            return family_templates.get(variant, family_templates[list(family_templates.keys())[0]])

        # 获取当前模型的模板
        template_info = get_model_template(self.model_name)

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            # 构建参考文献部分
            references = ""
            if len(retrieval_results) > 0:
                references += "# References \n"
                for snippet in retrieval_results:
                    references += f"- {snippet.strip()}\n"
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

            # 构建用户消息
            user_message = (
                f"{references}\n------\n\n"
                f"Using only the references listed above, answer the following question: \n"
                f"Current Time: {query_time}\n"
                f"Question: {query}\n"
            )

            if template_info["is_chat_model"]:
                try:
                    # 对于支持 chat template 的模型，尝试使用标准格式
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    # 如果失败，使用预定义的模板
                    formatted_prompt = template_info["template"].format(
                        system=system_prompt,
                        user=user_message
                    )
            else:
                # 对于基础模型，直接使用预定义的模板
                formatted_prompt = template_info["template"].format(
                    system=system_prompt,
                    user=user_message
                )
                
            formatted_prompts.append(formatted_prompt)

        return formatted_prompts

    def save_retrieval_history(self):
        """保存检索历史，使用dataset+embedding模型名称作为文件名"""
        # 构建文件名
        embedding_model_name = config.rag.embedding_model_name.split('/')[-1].lower()
        filename = f"{config.dataset.name}_{embedding_model_name}.json"
        filepath = os.path.join("predictions", "retrieval_history", filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            logger.info(f"Retrieval history file already exists at {filepath}, skipping save.")
            return
            
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存检索历史
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.retrieval_history, f, ensure_ascii=False, indent=2)
        logger.info(f"Retrieval history saved to {filepath}")
