from typing import List, Dict, Optional, Type
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class BaseDataProcessor(ABC):
    """Base class for dataset processors."""
    
    @abstractmethod
    def load_texts(
        self,
        data_path: Path,
        max_texts: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        """Load and process texts from dataset."""
        pass
    
    def filter_texts(
        self,
        texts: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        """Filter texts by length."""
        if min_length is not None:
            texts = [t for t in texts if len(t) >= min_length]
        if max_length is not None:
            texts = [t for t in texts if len(t) <= max_length]
        return texts

class JSONLProcessor(BaseDataProcessor):
    """Processor for JSONL format datasets."""
    
    def __init__(self, text_key: str = "text"):
        self.text_key = text_key
    
    def load_texts(
        self,
        data_path: Path,
        max_texts: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        texts = []
        
        with open(data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if text := data.get(self.text_key):
                        texts.append(text)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in file {data_path}")
                    continue
                
                if max_texts and len(texts) >= max_texts:
                    break
        
        return self.filter_texts(texts, min_length, max_length)

class CategoryProcessor(BaseDataProcessor):
    """Processor for category-specific datasets."""
    
    def __init__(self, category: str):
        self.category = category
        self.processors = {
            "persona": JSONLProcessor(text_key="statement"),
            "advanced-ai-risk": JSONLProcessor(text_key="question"),
            "sycophancy": JSONLProcessor(text_key="question"),
            "ethics": JSONLProcessor(text_key="scenario")
        }
    
    def load_texts(
        self,
        data_path: Path,
        max_texts: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        processor = self.processors.get(self.category)
        if not processor:
            raise ValueError(f"No processor found for category: {self.category}")
            
        return processor.load_texts(
            data_path=data_path,
            max_texts=max_texts,
            min_length=min_length,
            max_length=max_length
        )

def get_dataset_processor(categories: List[str]) -> BaseDataProcessor:
    """Factory function to get appropriate processor."""
    if len(categories) == 1:
        return CategoryProcessor(categories[0])
    
    # For multiple categories, create a composite processor
    class CompositeProcessor(BaseDataProcessor):
        def load_texts(self, *args, **kwargs) -> List[str]:
            all_texts = []
            for category in categories:
                processor = CategoryProcessor(category)
                texts = processor.load_texts(*args, **kwargs)
                all_texts.extend(texts)
            return all_texts
    
    return CompositeProcessor()

@dataclass
class DataSettings:
    """Settings for data preparation"""
    datasets: List[str]
    n_statements: int
    random_state: int = 42
