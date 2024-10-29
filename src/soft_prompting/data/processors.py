from typing import List, Dict, Optional, Type
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import glob

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
        # Map base categories to their text keys
        self.category_text_keys = {
            "persona": "statement",
            "advanced-ai-risk": "question",
            "sycophancy": "question",
            "winogenerated": "sentence_with_blank"
        }
    
    def load_texts(
        self,
        data_path: Path,
        max_texts: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        base_category = self.category.split('/')[0]
        text_key = self.category_text_keys.get(base_category)
        
        if not text_key:
            raise ValueError(f"No processor found for category: {base_category}")
        
        processor = JSONLProcessor(text_key=text_key)
        category_path = data_path / base_category
        
        if not category_path.exists():
            raise ValueError(f"Category path does not exist: {category_path}")
        
        # Handle specific file or subdirectory requests
        if '/' in self.category:
            specific_path = category_path / '/'.join(self.category.split('/')[1:])
            if specific_path.is_file() and specific_path.suffix == '.jsonl':
                # Single JSONL file
                logger.info(f"Processing specific JSONL file: {specific_path}")
                return processor.load_texts(
                    data_path=specific_path,
                    max_texts=max_texts,
                    min_length=min_length,
                    max_length=max_length
                )
            elif specific_path.is_dir():
                # Specific subdirectory
                logger.info(f"Processing specific subdirectory: {specific_path}")
                jsonl_files = glob.glob(str(specific_path / "**" / "*.jsonl"), recursive=True)
            else:
                raise ValueError(f"Invalid path specified: {specific_path}")
        else:
            # Process entire category
            logger.info(f"Processing entire category: {base_category}")
            jsonl_files = glob.glob(str(category_path / "**" / "*.jsonl"), recursive=True)
        
        if not jsonl_files:
            logger.warning(f"No JSONL files found in {self.category}")
            return []
            
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        all_texts = []
        for jsonl_path in jsonl_files:
            try:
                file_texts = processor.load_texts(
                    data_path=Path(jsonl_path),
                    max_texts=None,  # Don't limit individual files
                    min_length=min_length,
                    max_length=max_length
                )
                all_texts.extend(file_texts)
                logger.info(f"Loaded {len(file_texts)} texts from {jsonl_path}")
            except Exception as e:
                logger.warning(f"Error processing {jsonl_path}: {str(e)}")
                continue
        
        # Apply max_texts limit to combined texts
        if max_texts and len(all_texts) > max_texts:
            all_texts = all_texts[:max_texts]
        
        logger.info(f"Total texts loaded for {self.category}: {len(all_texts)}")
        return all_texts

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
