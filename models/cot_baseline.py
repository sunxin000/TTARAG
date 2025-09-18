# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from .rag_llama_baseline import RAGModel
from .config import config
from loguru import logger

class CoTModel(RAGModel):
    """
    Chain of Thought (CoT) baseline model that extends the RAG model
    by adding explicit reasoning steps in the prompt.
    """
    def __init__(self, do_tta=False):
        super().__init__(do_tta=do_tta)
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
        
    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats prompts with Chain of Thought reasoning template.
        """
        # 修改system prompt以适应通用问答场景
        system_prompt = (
            "You are an expert who answers questions by carefully reasoning through the available references. "
            "For each question, follow these steps:\n"
            "1. First, identify the key facts and information from the references that are relevant to the question\n"
            "2. Then, analyze how these facts help answer the question\n"
            "3. Finally, provide a clear and concise answer\n"
        )
        
        formatted_prompts = []
        template_info = self._get_model_template(self.model_name)
        
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            
            # 构建参考文献部分
            references = ""
            if len(retrieval_results) > 0:
                references += "# References \n"
                for snippet in retrieval_results:
                    references += f"- {snippet.strip()}\n"
            references = references[:config.rag.max_context_references_length]
            
            # 修改用户消息部分，简化 CoT 提示
            user_message = (
                f"{references}\n------\n\n"
                f"Question: {query}\n"
                f"Current Time: {query_time}\n\n"
                f"Answer this question using only the references above. Let's solve this step by step:\n"
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