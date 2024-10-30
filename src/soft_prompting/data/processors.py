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
    
    def _get_text_key(self, category_path: str) -> str:
        """Get the appropriate text key based on the category path."""
        parts = category_path.split('/')
        for part in parts:
            if part in self.category_text_keys:
                return self.category_text_keys[part]
        return "text"  # Default fallback
    
    def _process_jsonl_file(
        self,
        jsonl_path: Path,
        text_key: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        max_texts: Optional[int] = None
    ) -> List[str]:
        """Process a single JSONL file."""
        processor = JSONLProcessor(text_key=text_key)
        try:
            texts = processor.load_texts(
                data_path=jsonl_path,
                max_texts=max_texts,
                min_length=min_length,
                max_length=max_length
            )
            logger.info(f"Loaded {len(texts)} texts from {jsonl_path}")
            return texts
        except Exception as e:
            logger.warning(f"Error processing {jsonl_path}: {str(e)}")
            return []

    def load_texts(
        self,
        data_path: Path,
        max_texts: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        """Load texts from either a specific JSONL file or directory."""
        all_texts = []
        
        logger.info(f"Searching for category: {self.category}")
        logger.info(f"Base path: {data_path}")
        
        # Handle full path to JSONL file
        if self.category.endswith('.jsonl'):
            jsonl_path = data_path / self.category
            logger.info(f"Checking direct JSONL path: {jsonl_path}")
        else:
            # Try with .jsonl extension
            jsonl_path = data_path / f"{self.category}.jsonl"
            logger.info(f"Checking with .jsonl extension: {jsonl_path}")
            
            if not jsonl_path.exists():
                # Search in subdirectories
                logger.info("Searching in subdirectories...")
                for eval_dir in [d for d in data_path.iterdir() if d.is_dir()]:
                    potential_path = eval_dir / f"{self.category}.jsonl"
                    logger.info(f"Checking path: {potential_path}")
                    if potential_path.exists():
                        jsonl_path = potential_path
                        logger.info(f"Found file at: {jsonl_path}")
                        break
        
        if jsonl_path.exists():
            text_key = self._get_text_key(str(self.category))
            logger.info(f"Loading texts with key: {text_key}")
            texts = self._process_jsonl_file(
                jsonl_path,
                text_key,
                min_length,
                max_length,
                max_texts
            )
            # Print the actual texts being loaded
            for i, text in enumerate(texts):
                logger.info(f"\nText {i+1}:\n{text}\n")
            
            all_texts.extend(texts)
        else:
            logger.warning(f"No JSONL file found for category {self.category}")
            
        return all_texts

def discover_all_categories(data_path: Path) -> List[str]:
    """Discover all available JSONL files and convert to categories."""
    all_categories = []
    for eval_dir in data_path.iterdir():
        if not eval_dir.is_dir():
            continue
        # Find all JSONL files recursively
        for jsonl_file in eval_dir.rglob("*.jsonl"):
            # Convert path to category format
            rel_path = jsonl_file.relative_to(eval_dir)
            category = str(rel_path.parent / rel_path.stem)
            all_categories.append(category)
    return sorted(all_categories)

def get_dataset_processor(categories: List[str]) -> BaseDataProcessor:
    """Factory function to get appropriate processor."""
    
    # Handle "all" categories
    if len(categories) == 1 and categories[0] == "all":
        class AllCategoriesProcessor(BaseDataProcessor):
            def load_texts(
                self,
                data_path: Path,
                max_texts: Optional[int] = None,
                min_length: Optional[int] = None,
                max_length: Optional[int] = None
            ) -> List[str]:
                discovered_categories = discover_all_categories(data_path)
                logger.info(f"Discovered categories: {discovered_categories}")
                
                all_texts = []
                for category in discovered_categories:
                    processor = CategoryProcessor(category)
                    category_texts = processor.load_texts(
                        data_path=data_path,
                        max_texts=max_texts,
                        min_length=min_length,
                        max_length=max_length
                    )
                    all_texts.extend(category_texts)
                return all_texts
                
        return AllCategoriesProcessor()
    
    # For specific categories, use a single CategoryProcessor
    elif len(categories) == 1:
        return CategoryProcessor(categories[0])
    
    # For multiple categories, combine their results
    else:
        class MultiCategoryProcessor(BaseDataProcessor):
            def __init__(self, categories: List[str]):
                self.categories = categories
                
            def load_texts(
                self,
                data_path: Path,
                max_texts: Optional[int] = None,
                min_length: Optional[int] = None,
                max_length: Optional[int] = None
            ) -> List[str]:
                all_texts = []
                for category in self.categories:
                    processor = CategoryProcessor(category)
                    category_texts = processor.load_texts(
                        data_path=data_path,
                        max_texts=max_texts,
                        min_length=min_length,
                        max_length=max_length
                    )
                    all_texts.extend(category_texts)
                return all_texts
                
        return MultiCategoryProcessor(categories)

@dataclass
class DataSettings:
    """Settings for data preparation"""
    datasets: List[str]
    n_statements: int
    random_state: int = 42
