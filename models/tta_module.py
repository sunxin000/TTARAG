# models/tta_module.py
import torch
import copy
from typing import List, Tuple, Optional
from transformers import PreTrainedModel, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
# from FlagEmbedding import FlagReranker
from .config import config
from peft import (
    LoraConfig, 
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PromptEncoderConfig,
    PromptTuningConfig
)
import math
import torch.nn as nn
from loguru import logger

class BaseReranker:
    """Reranker基类"""
    def __init__(self, device: str):
        self.device = device
        
    def rerank(self, query: str, docs: List[str], top_k: Optional[int] = None) -> Tuple[List[str], List[float]]:
        raise NotImplementedError

class CrossEncoderReranker(BaseReranker):
    """基于CrossEncoder的Reranker"""
    def __init__(self, device: str):
        super().__init__(device)
        self.model = CrossEncoder(
            config.tta.cross_encoder_name,
            device=device
        )
        
    def rerank(self, query: str, docs: List[str], top_k: Optional[int] = None) -> Tuple[List[str], List[float]]:
        if top_k is None:
            top_k = len(docs)
            
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        
        doc_score_pairs = list(zip(docs, scores))
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        reranked_docs = [pair[0] for pair in sorted_pairs][:top_k]
        reranked_scores = [pair[1] for pair in sorted_pairs][:top_k]
        
        return reranked_docs, reranked_scores

# class BGEReranker(BaseReranker):
#     """基于BGE的Reranker"""
#     def __init__(self, device: str):
#         super().__init__(device)
#         self.model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
#     def rerank(self, query: str, docs: List[str], top_k: Optional[int] = None) -> Tuple[List[str], List[float]]:
#         if top_k is None:
#             top_k = len(docs)
            
#         qd_list = [[query, doc] for doc in docs]
#         scores = self.model.compute_score(qd_list)
        
#         doc_score_pairs = list(zip(docs, scores))
#         sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
#         reranked_docs = [pair[0] for pair in sorted_pairs][:top_k]
#         reranked_scores = [pair[1] for pair in sorted_pairs][:top_k]
        
#         return reranked_docs, reranked_scores

class TTAModule:
    def __init__(self, model: PreTrainedModel, tokenizer, learning_rate=config.tta.learning_rate):
        self.base_model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        
        # 保存原始状态用于全参数微调
        self._original_state = model.state_dict() if not config.tta.use_peft else None
        
        # 如果使用PEFT，根据method初始化对应配置
        if config.tta.use_peft:
            if config.tta.peft_method == "lora":
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config.tta.lora_r,
                    lora_alpha=config.tta.lora_alpha,
                    lora_dropout=config.tta.lora_dropout,
                    bias="none",
                    target_modules=["q_proj", "v_proj"]
                )
                self.base_model = get_peft_model(model, self.peft_config)
                
            elif config.tta.peft_method == "prefix":
                # 对于prefix tuning，确保所有配置都明确指定
                if config.tta.token_dim is None:
                    config.tta.token_dim = model.config.hidden_size
                if config.tta.num_transformer_submodules is None:
                    config.tta.num_transformer_submodules = 2
                if config.tta.num_attention_heads is None:
                    config.tta.num_attention_heads = model.config.num_attention_heads
                if config.tta.num_layers is None:
                    config.tta.num_layers = model.config.num_hidden_layers
                if config.tta.encoder_hidden_size is None:
                    config.tta.encoder_hidden_size = model.config.hidden_size

                self.peft_config = PrefixTuningConfig(
                    peft_type="PrefixTuning",
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=config.tta.num_virtual_tokens,
                    token_dim=config.tta.token_dim,
                    num_transformer_submodules=config.tta.num_transformer_submodules,
                    num_attention_heads=config.tta.num_attention_heads,
                    num_layers=config.tta.num_layers,
                    encoder_hidden_size=config.tta.encoder_hidden_size,
                    prefix_projection=config.tta.prefix_projection,
                    inference_mode=False,
                )
                
                # 先创建PEFT模型
                self.base_model = get_peft_model(model, self.peft_config)
                
            elif config.tta.peft_method == "ptuning":
                # 对于P-tuning，确保所有配置都明确指定
                if config.tta.token_dim is None:
                    config.tta.token_dim = model.config.hidden_size
                if config.tta.prompt_hidden_size is None:
                    config.tta.prompt_hidden_size = model.config.hidden_size

                # 创建P-tuning配置
                self.peft_config = PromptEncoderConfig(
                    peft_type="P_TUNING",
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=config.tta.num_virtual_tokens,
                    token_dim=config.tta.token_dim,
                    num_transformer_submodules=1,  # P-tuning只需要1个子模块
                    num_attention_heads=1,  # P-tuning不需要多头注意力
                    num_layers=config.tta.prompt_num_layers,
                    encoder_hidden_size=config.tta.prompt_hidden_size,
                    # encoder_rnn_type=config.tta.prompt_encoder_type,  # "lstm" or "mlp"
                    encoder_dropout=config.tta.prompt_dropout,
                )
                
                # 创建PEFT模型
                self.base_model = get_peft_model(model, self.peft_config)
                
        else:
            self.base_model = model
        
        # 初始化reranker
        if config.tta.use_sentence_selection:
            if config.tta.selection_method == "rerank": 
                reranker_type = config.tta.reranker_type
                if reranker_type == "cross_encoder":
                    self.reranker = CrossEncoderReranker(model.device)
                # elif reranker_type == "bge":
                #     self.reranker = BGEReranker(model.device)
                else:
                    raise ValueError(f"Unknown reranker type: {reranker_type}")
        
        if config.tta.use_dexperts:
            # Load expert model - will be adapted
            self.expert = AutoModelForCausalLM.from_pretrained(
                config.tta.expert_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            # Load antiexpert - same architecture but won't be adapted
            self.antiexpert = AutoModelForCausalLM.from_pretrained(
                config.tta.expert_model_name,  # Use same model as expert
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            self._expert_original_state = self.expert.state_dict()
            self.expert.eval()
            self.antiexpert.eval()
            self.alpha = config.tta.alpha
        # 初始化sentence selection相关组件
        if config.tta.use_sentence_selection:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(
                config.tta.cross_encoder_name,
                device=model.device
            )

    def _reset_model(self):
        """Reset model parameters based on configuration"""
        if config.tta.use_peft:
            if isinstance(self.base_model, PeftModel):
                if config.tta.peft_method == "lora":
                    # 重置LoRA权重
                    for name, param in self.base_model.named_parameters():
                        if 'lora_' in name:
                            if 'lora_A' in name:
                                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                            elif 'lora_B' in name:
                                nn.init.zeros_(param)
                elif config.tta.peft_method == "prefix":
                    # 重置prefix embedding参数
                    for name, param in self.base_model.named_parameters():
                        if 'prefix_encoder' in name:
                            if 'embedding' in name:
                                # 初始化embedding
                                nn.init.normal_(param, mean=0.0, std=0.02)
                            elif 'encoder' in name:
                                # 初始化transformer权重
                                if 'weight' in name:
                                    nn.init.normal_(param, mean=0.0, std=0.02)
                                elif 'bias' in name:
                                    nn.init.zeros_(param)
                elif config.tta.peft_method == "ptuning":
                    # 重置P-tuning参数
                    for name, param in self.base_model.named_parameters():
                        if 'prompt_encoder' in name:
                            if 'embedding' in name:
                                # 初始化embedding
                                nn.init.normal_(param, mean=0.0, std=0.02)
                            elif config.tta.prompt_encoder_type == "lstm":
                                # LSTM权重初始化
                                if 'weight' in name:
                                    nn.init.orthogonal_(param)
                                elif 'bias' in name:
                                    nn.init.zeros_(param)
                            else:  # MLP
                                if 'weight' in name:
                                    nn.init.normal_(param, mean=0.0, std=0.02)
                                elif 'bias' in name:
                                    nn.init.zeros_(param)
        else:
            # 使用原始的全参数重置逻辑
            if config.tta.use_dexperts:
                self.expert.load_state_dict(self._expert_original_state)
            else:
                self.base_model.load_state_dict(self._original_state)

    def _adapt_model(self, model, adapt_pairs, wosegment=False):
        """Adapt model based on configuration"""
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            eps=config.tta.adam_epsilon,
            weight_decay=config.tta.weight_decay
        )
        
        accumulation_steps = config.tta.accumulation_steps
        optimizer.zero_grad()
        
        for i, (prefix, suffix) in enumerate(adapt_pairs):
            full_text = prefix + suffix
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)
            
            labels = inputs["input_ids"].clone()
            if not wosegment:  # Only set prefix labels to -100 if wosegment is False
                prefix_tokens = self.tokenizer(
                    prefix,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                prefix_len = prefix_tokens["input_ids"].size(1)
                labels[0, :prefix_len] = -100
            
            logger.info(f"model_device: {model.device}")
            # logger.info(f"inputs: {inputs}")
            outputs = model(**inputs, labels=labels)
            
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                logger.error(f"Loss is NaN or Inf for pair: {prefix} {suffix}")
                return False
                
            loss = outputs.loss / (len(adapt_pairs) * accumulation_steps)
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.tta.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                
        if len(adapt_pairs) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        model.eval()
        return True

    def _get_dexperts_logits(self, input_ids, attention_mask=None):
        """Get logits from expert and antiexpert models"""
        with torch.no_grad():
            expert_outputs = self.expert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            antiexpert_outputs = self.antiexpert(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True
            )
            
            expert_logits = expert_outputs.logits[..., -1, :]
            antiexpert_logits = antiexpert_outputs.logits[..., -1, :]
            
            # Ensure same vocabulary size
            min_vocab_size = min(expert_logits.shape[-1], antiexpert_logits.shape[-1])
            expert_logits = expert_logits[..., :min_vocab_size]
            antiexpert_logits = antiexpert_logits[..., :min_vocab_size]
            
            return expert_logits, antiexpert_logits

    def adapt_model(self, context_sentences: List[str], query: str = None, wosegment: bool = False) -> Optional[PreTrainedModel]:
        """
        根据上下文句子调整模型
        
        Parameters:
            context_sentences (List[str]): 上下文句子列表
            query (str, optional): 当前查询,用于sentence selection
            wosegment (bool, optional): 是否不对prefix部分的label设置为-100
        """
        # 首先进行sentence selection
        if query and config.tta.use_sentence_selection:
            context_sentences = self.select_valuable_sentences(query, context_sentences)
            
        # 准备adaptation数据
        adapt_pairs = self.prepare_adaptation_data(context_sentences)
        if not adapt_pairs:
            return None
            
        torch.cuda.empty_cache()
        adapt_pairs = adapt_pairs[:config.tta.max_adapt_pairs]
        
        # try:
        self._reset_model()
        
        if config.tta.use_dexperts:
            # Adapt expert model when using DExperts
            if not self._adapt_model(self.expert, adapt_pairs, wosegment):
                return None
        else:
            # Adapt base model when using traditional TTA
            if not self._adapt_model(self.base_model, adapt_pairs, wosegment):
                return None
            
        return self.base_model
            
        # except Exception as e:
        #     print(f"Error during adaptation: {str(e)}")
        #     return None

    def prepare_adaptation_data(self, sentences: List[str]) -> List[Tuple[str, str]]:
        """准备用于适应的前缀-后缀对."""
        pairs = []
        for sent in sentences:
            # 去除问句
            if len(sent.split()) >= config.tta.min_sentence_length: # and sent[-1] != "?":
                prefix, suffix = self.split_sentence(sent)
                pairs.append((prefix, suffix))
        return pairs

    def split_sentence(self, sentence: str) -> Tuple[str, str]:
        """在标点或中间位置分割句子."""
        punctuations = ['.', ',', ';', ':', '!', '?']
        
        # 先尝试在标点处分割
        for punct in punctuations:
            if punct in sentence:
                parts = sentence.split(punct, 1)
                if len(parts[0].split()) >= 3 and len(parts[1].split()) >= 3:
                    return parts[0] + punct, parts[1]
        
        # 如果没有合适的标点，在中间分割
        words = sentence.split()
        mid = len(words) // 2
        return ' '.join(words[:mid]), ' '.join(words[mid:])

    def select_valuable_sentences(self, query: str, sentences: List[str]) -> List[str]:
        """
        选择对TTA最有价值的句子
        
        Parameters:
            query (str): 当前的查询
            sentences (List[str]): 候选句子列表
            
        Returns:
            List[str]: 筛选后的句子列表
        """
        if not config.tta.use_sentence_selection:
            return sentences
            
        if config.tta.selection_method == "rerank":
            # 使用reranker重排序
            reranked_sents, _ = self.reranker.rerank(
                query,
                sentences,
                top_k=config.tta.max_selected_sentences
            )
            return reranked_sents
            
        elif config.tta.selection_method == "cluster":
            # 使用聚类方法选择有代表性的句子
            from sklearn.cluster import KMeans
            from sentence_transformers import SentenceTransformer
            
            # 获取句子embeddings
            encoder = SentenceTransformer(config.tta.embedding_model_name)
            embeddings = encoder.encode(sentences)
            
            # 进行聚类
            n_clusters = min(config.tta.max_selected_sentences, len(sentences))
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(embeddings)
            
            # 从每个簇中选择最接近中心的句子
            selected = []
            for i in range(n_clusters):
                cluster_mask = clusters == i
                if not any(cluster_mask):
                    continue
                    
                cluster_embeddings = embeddings[cluster_mask]
                cluster_sentences = [s for s, m in zip(sentences, cluster_mask) if m]
                
                # 计算到簇中心的距离
                distances = ((cluster_embeddings - kmeans.cluster_centers_[i]) ** 2).sum(axis=1)
                selected.append(cluster_sentences[distances.argmin()])
                
        elif config.tta.selection_method == "summarize":
            # 使用摘要模型压缩信息
            from transformers import pipeline
            
            summarizer = pipeline(
                "summarization",
                model=config.tta.summarizer_name,
                device=self.base_model.device
            )
            
            # 将句子组合成段落
            text = " ".join(sentences)
            
            # 生成摘要
            summary = summarizer(
                text,
                max_length=config.tta.max_summary_length,
                min_length=config.tta.min_summary_length,
                do_sample=False
            )[0]["summary_text"]
            
            # 将摘要分割成句子
            from nltk.tokenize import sent_tokenize
            selected = sent_tokenize(summary)
            
        else:
            selected = sentences
            
        return selected