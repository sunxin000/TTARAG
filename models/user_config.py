# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# isort: skip_file
# from models.dummy_model import DummyModel
# UserModel = DummyModel

# Uncomment the lines below to use the Vanilla LLAMA baseline
# from models.vanilla_llama_baseline import InstructModel 
# UserModel = InstructModel


# Uncomment the lines below to use the RAG LLAMA baseline
from models.rag_llama_baseline import RAGModel
# from models.cot_baseline import CoTModel
# from models.icl_baseline import ICLModel

# Choose which model to use as UserModel
UserModel = RAGModel  # or RAGModel or ICLModel

# Uncomment the lines below to use the RAG KG LLAMA baseline
# from models.rag_knowledge_graph_baseline import RAG_KG_Model
# UserModel = RAG_KG_Model
