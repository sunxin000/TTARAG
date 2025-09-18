from pydantic import BaseModel, Field
from typing import Dict
import argparse

class TTAConfig(BaseModel):
    learning_rate: float = 1e-5
    max_adapt_pairs: int = 3
    min_sentence_length: int = 6
    accumulation_steps: int = 2
    max_grad_norm: float = 0.1
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    use_dexperts: bool = False
    expert_model_name:str = "xxxxx"
    antiexpert_model_name: str = "xxxxx"
    alpha: float = 1.0
    use_sentence_selection: bool = False
    selection_method: str = "rerank"  # 可选: "rerank", "cluster", "summarize"
    max_selected_sentences: int = 10
    cross_encoder_name: str = "models/ms-marco-MiniLM-L-6-v2"
    summarizer_name: str = "facebook/bart-large-cnn"
    max_summary_length: int = 150
    min_summary_length: int = 50
    reranker_type: str = "cross_encoder"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    use_peft: bool = False
    peft_method: str = "lora"  # 可选: "lora", "prefix"
    num_virtual_tokens: int = 20    # 虚拟tokens数量
    token_dim: int = None          # token维度，如果为None则使用hidden_size
    num_transformer_submodules: int = None  # transformer子模块数量
    num_attention_heads: int = None  # 注意力头数量
    num_layers: int = None          # 层数
    encoder_hidden_size: int = None  # encoder隐藏层大小
    prefix_projection: bool = True   # 是否使用prefix projection
    #p-tunning 配置
    prompt_encoder_type: str = "mlp"  # 可选: "lstm", "mlp"
    prompt_num_layers: int = 2  # prompt encoder层数
    prompt_dropout: float = 0.1  # prompt encoder dropout
    prompt_hidden_size: int = None 
    wosegment: bool = False

class GenerationParams(BaseModel):
    max_new_tokens: int = 200
    temperature: float = 0.1
    top_p: float = 0.9

class RAGConfig(BaseModel):
    num_context_sentences: int = 20
    max_context_sentence_length: int = 1000
    max_context_references_length: int = 4000
    submission_batch_size: int = 1
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85
    sentence_transformer_batch_size: int = 128
    generation_params: GenerationParams = GenerationParams()
    model_name: str = "models/Llama-3.1-8B-Instruct"
    embedding_model_name: str = "models/allMiniLM"

class ParallelEvalConfig(BaseModel):
    default_max_workers: int = 10

class DatasetConfig(BaseModel):
    name: str = "crag"  # or "crag"
    split: str = "train"    # for pubmedqa
    path: str = None        # for crag

# Add dataset config to main config
class Config(BaseModel):
    tta: TTAConfig = TTAConfig()
    rag: RAGConfig = RAGConfig()
    parallel_eval: ParallelEvalConfig = ParallelEvalConfig()
    dataset: DatasetConfig = DatasetConfig()

def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """根据命令行参数更新配置，只更新明确指定的非None值"""
    
    # RAG 配置
    if hasattr(args, 'model_name') and args.model_name is not None:
        config.rag.model_name = args.model_name
        
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.rag.submission_batch_size = args.batch_size
        
    if hasattr(args, 'tensor_parallel_size') and args.tensor_parallel_size is not None:
        config.rag.vllm_tensor_parallel_size = args.tensor_parallel_size
        
    if hasattr(args, 'gpu_memory_utilization') and args.gpu_memory_utilization is not None:
        config.rag.vllm_gpu_memory_utilization = args.gpu_memory_utilization
    
    # TTA 配置
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config.tta.learning_rate = args.learning_rate
        
    if hasattr(args, 'max_adapt_pairs') and args.max_adapt_pairs is not None:
        config.tta.max_adapt_pairs = args.max_adapt_pairs
        
    if hasattr(args, 'use_dexperts'):  # boolean flag 不需要检查 None
        config.tta.use_dexperts = args.use_dexperts
        
    if hasattr(args, 'expert_model_name') and args.expert_model_name is not None:
        config.tta.expert_model_name = args.expert_model_name
        
    if hasattr(args, 'alpha') and args.alpha is not None:
        config.tta.alpha = args.alpha
    
    if hasattr(args, 'reranker_type') and args.reranker_type is not None:
        config.tta.reranker_type = args.reranker_type   
        
    if hasattr(args, 'max_selected_sentences') and args.max_selected_sentences is not None:
        config.tta.max_selected_sentences = args.max_selected_sentences     
        
    if hasattr(args, 'cross_encoder_name') and args.cross_encoder_name is not None:
        config.tta.cross_encoder_name = args.cross_encoder_name
        
    if hasattr(args, 'summarizer_name') and args.summarizer_name is not None:
        config.tta.summarizer_name = args.summarizer_name   
        
    if hasattr(args, 'max_summary_length') and args.max_summary_length is not None:
        config.tta.max_summary_length = args.max_summary_length
        
    if hasattr(args, 'min_summary_length') and args.min_summary_length is not None:
        config.tta.min_summary_length = args.min_summary_length   
    
    if hasattr(args, 'use_sentence_selection'):
        config.tta.use_sentence_selection = args.use_sentence_selection
        
    if hasattr(args, 'lora_r') and args.lora_r is not None:
        config.tta.lora_r = args.lora_r
        
    if hasattr(args, 'lora_alpha') and args.lora_alpha is not None:
        config.tta.lora_alpha = args.lora_alpha
        
    if hasattr(args, 'lora_dropout') and args.lora_dropout is not None:
        config.tta.lora_dropout = args.lora_dropout
    
    if hasattr(args, 'use_peft'):
        config.tta.use_peft = args.use_peft
        
    # 数据集配置
    if hasattr(args, 'dataset') and args.dataset is not None:
        config.dataset.name = args.dataset
        
    if hasattr(args, 'dataset_path') and args.dataset_path is not None:
        config.dataset.path = args.dataset_path
        
    # 并行评估配置
    if hasattr(args, 'max_workers') and args.max_workers is not None:
        config.parallel_eval.default_max_workers = args.max_workers
    
    # PEFT相关配置
    if hasattr(args, 'peft_method') and args.peft_method is not None:
        config.tta.peft_method = args.peft_method
        
    # LoRA配置
    if hasattr(args, 'lora_r') and args.lora_r is not None:
        config.tta.lora_r = args.lora_r
        
    if hasattr(args, 'lora_alpha') and args.lora_alpha is not None:
        config.tta.lora_alpha = args.lora_alpha
        
    if hasattr(args, 'lora_dropout') and args.lora_dropout is not None:
        config.tta.lora_dropout = args.lora_dropout
        
    # Prefix Tuning配置
    if hasattr(args, 'num_virtual_tokens') and args.num_virtual_tokens is not None:
        config.tta.num_virtual_tokens = args.num_virtual_tokens
        
    if hasattr(args, 'token_dim') and args.token_dim is not None:
        config.tta.token_dim = args.token_dim
        
    if hasattr(args, 'num_transformer_submodules') and args.num_transformer_submodules is not None:
        config.tta.num_transformer_submodules = args.num_transformer_submodules
        
    if hasattr(args, 'num_attention_heads') and args.num_attention_heads is not None:
        config.tta.num_attention_heads = args.num_attention_heads
        
    if hasattr(args, 'num_layers') and args.num_layers is not None:
        config.tta.num_layers = args.num_layers
        
    if hasattr(args, 'encoder_hidden_size') and args.encoder_hidden_size is not None:
        config.tta.encoder_hidden_size = args.encoder_hidden_size
        
    if hasattr(args, 'prefix_projection') and args.prefix_projection is not None:
        config.tta.prefix_projection = args.prefix_projection
    
    if hasattr(args, 'use_peft'):
        config.tta.use_peft = args.use_peft
    
    return config

# 创建全局配置实例
config = Config() 