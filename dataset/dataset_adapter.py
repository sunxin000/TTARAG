from datasets import load_dataset
from typing import Dict, List, Any, Tuple
import json
import bz2
import logging
import os
import tqdm

logger = logging.getLogger(__name__)

corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedText": ["textbooks", "statpearls"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
}

class DocExtracter:
    """Extract document text from document IDs"""
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedCorp"):
        self.db_dir = db_dir
        self.cache = cache
        logger.info("Initializing the document extracter...")
        
        # Initialize corpus directories
        for corpus in corpus_names[corpus_name]:
            if not os.path.exists(os.path.join(self.db_dir, corpus, "chunk")):
                logger.info(f"Cloning the {corpus} corpus from Huggingface...")
                os.system(f"git clone https://huggingface.co/datasets/MedRAG/{corpus} {os.path.join(self.db_dir, corpus)}")
                if corpus == "statpearls":
                    logger.info("Downloading the statpearls corpus from NCBI bookshelf...")
                    os.system(f"wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {os.path.join(self.db_dir, corpus)}")
                    os.system(f"tar -xzvf {os.path.join(self.db_dir, corpus, 'statpearls_NBK430685.tar.gz')} -C {os.path.join(self.db_dir, corpus)}")
                    logger.info("Chunking the statpearls corpus...")
                    os.system("python src/data/statpearls.py")
        
        # Load or create cache/path mapping
        if self.cache:
            cache_file = os.path.join(self.db_dir, f"{corpus_name}_id2text.json")
            if os.path.exists(cache_file):
                self.dict = json.load(open(cache_file))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        chunk_path = os.path.join(self.db_dir, corpus, "chunk", fname)
                        if open(chunk_path).read().strip() == "":
                            continue
                        for line in open(chunk_path).read().strip().split('\n'):
                            item = json.loads(line)
                            _ = item.pop("contents", None)
                            self.dict[item["id"]] = item
                with open(cache_file, 'w') as f:
                    json.dump(self.dict, f)
        else:
            path_file = os.path.join(self.db_dir, f"{corpus_name}_id2path.json")
            if os.path.exists(path_file):
                self.dict = json.load(open(path_file))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        chunk_path = os.path.join(self.db_dir, corpus, "chunk", fname)
                        if open(chunk_path).read().strip() == "":
                            continue
                        for i, line in enumerate(open(chunk_path).read().strip().split('\n')):
                            item = json.loads(line)
                            self.dict[item["id"]] = {"fpath": os.path.join(corpus, "chunk", fname), "index": i}
                with open(path_file, 'w') as f:
                    json.dump(self.dict, f, indent=4)
        logger.info("Document extracter initialization finished!")
    
    def extract(self, ids):
        """Extract document text from document IDs"""
        if self.cache:
            output = []
            for i in ids:
                item = self.dict[i] if isinstance(i, str) else self.dict[i["id"]]
                output.append(item)
        else:
            output = []
            for i in ids:
                item = self.dict[i] if isinstance(i, str) else self.dict[i["id"]]
                output.append(json.loads(open(os.path.join(self.db_dir, item["fpath"])).read().strip().split('\n')[item["index"]]))
        return output

class DatasetAdapter:
    """Base class for dataset adapters"""
    def load_data(self):
        raise NotImplementedError
        
    def format_sample(self, sample):
        raise NotImplementedError

    def load_data_in_batches(self, batch_size):
        """Load data in batches"""
        data = self.load_data()
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            yield {
                "interaction_id": [self.format_sample(sample)["interaction_id"] for sample in batch],
                "query": [self.format_sample(sample)["query"] for sample in batch],
                "search_results": [self.format_sample(sample)["search_results"] for sample in batch],
                "query_time": [self.format_sample(sample)["query_time"] for sample in batch],
                "answer": [self.format_sample(sample)["ground_truth"] for sample in batch]
            }

class CRAGAdapter(DatasetAdapter):
    """Adapter for CRAG dataset"""
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def load_data(self):
        with bz2.open(self.dataset_path, "rt") as f:
            data = [json.loads(line) for line in f]
        return data
        
    def format_sample(self, sample) -> Dict[str, Any]:
        return {
            "interaction_id": sample["interaction_id"],
            "query": sample["query"],
            "search_results": sample["search_results"],
            "query_time": sample["query_time"],
            "ground_truth": sample["answer"],
            "domain": sample.get("domain", "unknown"),
            "question_type": sample.get("question_type", "unknown"),
            "static_or_dynamic": sample.get("static_or_dynamic", "unknown")
        }

    def load_data_in_batches(self, batch_size):
        """Override for CRAG specific batch loading"""
        def initialize_batch():
            return {
                "interaction_id": [], 
                "query": [], 
                "search_results": [], 
                "query_time": [], 
                "answer": [],
                "domain": [],
                "question_type": [],
                "static_or_dynamic": []
            }
            
        try:
            with bz2.open(self.dataset_path, "rt") as file:
                batch = initialize_batch()
                for line in file:
                    try:
                        item = json.loads(line)
                        batch["interaction_id"].append(item["interaction_id"])
                        batch["query"].append(item["query"])
                        batch["search_results"].append(item["search_results"])
                        batch["query_time"].append(item["query_time"])
                        batch["answer"].append(item["answer"])
                        batch["domain"].append(item.get("domain", "unknown"))
                        batch["question_type"].append(item.get("question_type", "unknown"))
                        batch["static_or_dynamic"].append(item.get("static_or_dynamic", "unknown"))
                        
                        if len(batch["query"]) == batch_size:
                            yield batch
                            batch = initialize_batch()
                    except json.JSONDecodeError:
                        logger.warn("Warning: Failed to decode a line.")
                # Yield any remaining data as the last batch
                if batch["query"]:
                    yield batch
        except FileNotFoundError as e:
            logger.error(f"Error: The file {self.dataset_path} was not found.")
            raise e
        except IOError as e:
            logger.error(f"Error: An error occurred while reading the file {self.dataset_path}.")
            raise e

class PubMedQAAdapter(DatasetAdapter):
    """Adapter for PubMedQA dataset"""
    def __init__(self, split="train"):
        self.split = split
        self.dataset = None
        
    def load_data(self):
        if self.dataset is None:
            self.dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")['train']
        return self.dataset
        
    def format_sample(self, sample) -> Dict[str, Any]:
        # Format context as search results
        contexts = sample["context"]["contexts"] if isinstance(sample["context"], dict) else eval(sample["context"])["contexts"]
        search_results = [{"page_result": ctx} for ctx in contexts]
        
        return {
            "interaction_id": str(sample["pubid"]),
            "query": sample["question"],
            "search_results": search_results,
            "query_time": "2024-03-21 00:00:00",
            "ground_truth": sample["final_decision"],
            "domain": "medical"  # PubMedQA is always medical domain
        }

    def load_data_in_batches(self, batch_size):
        """Override for PubMedQA specific batch loading to handle HuggingFace dataset batching"""
        if self.dataset is None:
            self.dataset = self.load_data()
            
        # HuggingFace datasets already handles batching efficiently
        for i in range(0, len(self.dataset), batch_size):
            batch_data = self.dataset[i:i + batch_size]
            
            # Process each field separately since batch_data is column-oriented
            formatted_batch = {
                "interaction_id": [],
                "query": [],
                "search_results": [],
                "query_time": [],
                "answer": [],
                "domain": []  # Add domain field
            }
            
            # Iterate through indices to maintain alignment
            for idx in range(len(batch_data['pubid'])):
                # Create a sample dictionary with aligned data
                sample = {
                    'pubid': batch_data['pubid'][idx],
                    'question': batch_data['question'][idx],
                    'context': batch_data['context'][idx],
                    'final_decision': batch_data['final_decision'][idx]
                }
                
                # Format the sample and add to batch
                formatted_sample = self.format_sample(sample)
                formatted_batch["interaction_id"].append(formatted_sample["interaction_id"])
                formatted_batch["query"].append(formatted_sample["query"])
                formatted_batch["search_results"].append(formatted_sample["search_results"])
                formatted_batch["query_time"].append(formatted_sample["query_time"])
                formatted_batch["answer"].append(formatted_sample["ground_truth"])
                formatted_batch["domain"].append(formatted_sample["domain"])
            
            yield formatted_batch

class MedRAGAdapter(DatasetAdapter):
    """Adapter for MedRAG dataset"""
    def __init__(self, split="test", data_dir="/ossfs/workspace/CRAG", corpus_db_dir="./corpus"):
        self.split = split
        self.data_dir = data_dir
        self.benchmark = json.load(open(os.path.join(data_dir, "benchmark.json")))
        self.corpus_db_dir = corpus_db_dir
        self.doc_extractor = DocExtracter(db_dir=corpus_db_dir, cache=True, corpus_name="MedCorp")
        
    def load_data(self):
        # Load data from benchmark.json without removing options
        data = []
        # Only include pubmedqa and bioasq datasets
        allowed_datasets = {"pubmedqa", "bioasq"}
        
        for dataset_name, dataset in self.benchmark.items():
            if dataset_name.lower() in allowed_datasets:
                for qid, item in dataset.items():
                    data.append({
                        "dataset": dataset_name,
                        "qid": qid,
                        "question": item["question"],
                        "options": item["options"],
                        "answer": item["answer"]
                    })
        return data
        
    def format_sample(self, sample) -> Dict[str, Any]:
        # Load retrieved snippets
        retrieval_path = os.path.join(
            self.data_dir,
            "retrieved_snippets_10k",
            sample["dataset"],
            "wikipedia/facebook/contriever/snippets",
            f"{self.split}_{sample['qid']}.json"
        )
        
        if not os.path.exists(retrieval_path):
            search_results = []
        else:
            with open(retrieval_path) as f:
                retrieved_data = json.load(f)
                # Extract documents and format them as search results
                docs = self.doc_extractor.extract(retrieved_data)
                search_results = [{"page_result": doc["content"]} for doc in docs]
        
        # Format question with options
        options_text = '\n'.join([f"{key}. {sample['options'][key]}" for key in sorted(sample['options'].keys())])
        query = f"{sample['question']}\n\n{options_text}"
            
        return {
            "interaction_id": f"{sample['dataset']}_{sample['qid']}",
            "query": query,
            "search_results": search_results,
            "query_time": "2024-03-21 00:00:00",
            "ground_truth": sample["answer"],
            "domain": sample["dataset"].lower(),  # Use dataset name as domain
        }

def get_dataset_adapter(dataset_name: str, **kwargs) -> DatasetAdapter:
    """Factory function to get appropriate dataset adapter"""
    adapters = {
        "crag": CRAGAdapter,
        "pubmedqa": PubMedQAAdapter,
        "medrag": MedRAGAdapter
    }
    
    adapter_class = adapters.get(dataset_name.lower())
    if not adapter_class:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return adapter_class(**kwargs) 