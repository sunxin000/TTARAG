# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from .rag_llama_baseline import RAGModel
from .config import config
from loguru import logger
import json
import os

class ICLModel(RAGModel):
    """
    In-Context Learning (ICL) baseline model that extends the RAG model
    by incorporating few-shot examples in the prompt.
    """
    def __init__(self, do_tta=False, num_examples=3):
        super().__init__(do_tta=do_tta)
        self.num_examples = num_examples
        self.examples = self._load_examples()
        # 从RAG模型继承模板
        self.MODEL_TEMPLATES = {
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
        
    def _load_examples(self):
        """
        Load few-shot examples from a JSON file.
        Returns a list of examples in the format:
        [{"query": "...", "references": "...", "answer": "..."}, ...]
        """
        examples_path = os.path.join("data", f"{config.dataset.name}_examples.json")
        try:
            with open(examples_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            logger.info(f"Loaded {len(examples)} examples from {examples_path}")
            return examples[:self.num_examples]  # 只使用指定数量的示例
        except FileNotFoundError:
            logger.warning(f"Examples file not found at {examples_path}, using default examples")
            return self._get_default_examples()
            
    def _get_default_examples(self):
        """提供默认的few-shot示例"""
        if config.dataset.name == "crag":
            return [
                {
                    "query": "is microsoft office 2019 available in a greater number of languages than microsoft office 2013?",
                    "references": "- Currently Office 2010, 2013, 2016 and 2019 are supported\n- If you're an administrator who has deployed a volume licensed version of Office 2019 to your users, you can deploy language packs\n- Office 2019 requires Windows 10 1809, and cannot run on Windows 7\n",
                    "answer": "Yes."
                },
                {
                    "query": "how many times has rory mcilroy won the masters tournament?",
                    "references": "- McIlroy has won four major tournaments.\n- McIlroy has yet to win the Masters, though he has come close on several occasions.\n- McIlroy's best tournaments overall are the PGA Championship and the Masters.\n",
                    "answer": "0"
                },
                {
                    "query": "is dreamworks animation owned by time warner or universal pictures?",
                    "references": "- In April 2016, Comcast (which owns Universal) purchased DreamWorks Animation for $3.8 Billion.\n- As of October 2016, DreamWorks' films are marketed and distributed by Universal Pictures.\n",
                    "answer": "Universal Pictures."
                }
            ]
        else:  # medrag dataset
            return [
                {
                    "query": "A 45-year-old patient presents with chest pain and shortness of breath. ECG shows ST elevation in leads V1-V4. What is the most appropriate immediate treatment?",
                    "references": "- Primary PCI is the preferred reperfusion strategy for STEMI when available within 90 minutes\n- Thrombolysis should be considered if PCI cannot be performed within 120 minutes\n- Aspirin and anticoagulation should be administered immediately\n",
                    "options": {
                        "A": "Oral beta blockers",
                        "B": "Primary PCI",
                        "C": "Oral nitrates",
                        "D": "CT coronary angiogram"
                    },
                    "answer": "B"
                },
                {
                    "query": "Which medication is most commonly associated with angioedema in patients taking ACE inhibitors?",
                    "references": "- ACE inhibitors can cause angioedema in 0.1-0.7% of patients\n- The risk is highest in the first month of treatment\n- African Americans have a 3-4 times higher risk of ACE inhibitor-induced angioedema\n",
                    "options": {
                        "A": "Lisinopril",
                        "B": "Amlodipine",
                        "C": "Metoprolol",
                        "D": "Hydrochlorothiazide"
                    },
                    "answer": "A"
                }
            ]
    
    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats prompts with in-context examples.
        """
        system_prompt = (
            "You are an expert who answers questions based on provided references. "
            "Below are some example Q&A pairs followed by a new question. "
            "For multiple choice questions, respond with just the letter (A, B, C, or D) of the correct answer. "
            "For other questions, answer using the fewest words possible while maintaining accuracy. "
            "Base your answer only on the provided references. "
            "There is no need to explain your reasoning."
        )
        
        formatted_prompts = []
        template_info = self._get_model_template(self.model_name)
        
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            
            # 构建few-shot示例部分，使用更简洁的格式
            examples_text = "# Examples\n\n"
            for example in self.examples:
                examples_text += f"References:\n{example['references']}\n"
                examples_text += f"Question: {example['query']}\n"
                
                # 如果是选择题，添加选项
                if 'options' in example:
                    examples_text += "Options:\n"
                    for opt_key, opt_value in example['options'].items():
                        examples_text += f"{opt_key}: {opt_value}\n"
                
                examples_text += f"Answer: {example['answer']}\n\n"
            
            # 构建当前问题的参考文献
            current_references = ""
            if len(retrieval_results) > 0:
                current_references += "# Current References \n"
                for snippet in retrieval_results:
                    current_references += f"- {snippet.strip()}\n"
            current_references = current_references[:config.rag.max_context_references_length]
            
            # 构建用户消息
            user_message = (
                f"{examples_text}\n"
                f"Now, please answer the following question using only the references provided:\n"
                f"{current_references}\n------\n\n"
                f"Current Time: {query_time}\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            
            if template_info["is_chat_model"]:
                try:
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
                    formatted_prompt = template_info["template"].format(
                        system=system_prompt,
                        user=user_message
                    )
            else:
                formatted_prompt = template_info["template"].format(
                    system=system_prompt,
                    user=user_message
                )
                
            formatted_prompts.append(formatted_prompt)
            
        return formatted_prompts

    def _get_model_template(self, model_name):
        """Helper method to get the appropriate template for the model"""
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
            return self.MODEL_TEMPLATES["llama"]["base"]
        
        # 确定是否是 instruct 版本
        variant = "instruct" if any(x in model_name for x in ["-it", "instruct", "-i-"]) else "base"
        
        # 获取模板
        family_templates = self.MODEL_TEMPLATES[family]
        return family_templates.get(variant, family_templates[list(family_templates.keys())[0]]) 