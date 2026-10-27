# Specific Examples of Duplicated Code

## 1. API Request Handling

### Example A: behavioural_clustering/models/api_models.py
```python
class OpenAIModel:
    def generate(self, prompt, max_tokens=None):
        @retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(model, prompt, max_tokens):
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                timeout=60,
            )
            return completion
```

### Example B: interventions/auto_finetune_eval/auto_finetuning_helpers.py
```python
def make_api_request(prompt, api_provider, model_str, api_key=None, client=None, ...):
    if api_provider == "openai":
        if client is None:
            client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_str,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content
```

**Duplication**: Both implement OpenAI API calls with retry logic, but with different interfaces and error handling.

## 2. Embedding Generation

### Example A: behavioural_clustering/utils/embedding_utils.py
```python
def embed_texts(texts: List[str], embedding_settings):
    client = OpenAI()
    embeddings = []
    n_batches = n_texts // batch_size + int(n_texts % batch_size != 0)
    
    for i in tqdm(range(n_batches)):
        for retry_count in range(max_retries):
            try:
                embeddings_data = client.embeddings.create(
                    model=embedding_model, input=text_subset
                )
                break
            except Exception as e:
                time.sleep(initial_sleep_time * (2**retry_count))
```

### Example B: Similar pattern needed in contrastive-decoding but uses local models
```python
# In quick_cluster.py
parser.add_argument("--local_embedding_model_str", default="nvidia/NV-Embed-v1")
# Direct model usage without unified interface
```

**Duplication**: Different embedding approaches without shared abstraction.

## 3. Data Loading from JSONL

### Example A: behavioural_clustering/utils/data_preparation.py
```python
def _load_jsonl_file(self, file_path: Path) -> Tuple[List[Tuple[str, str]], bool]:
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "statement" in data:
                        texts.append((str(file_path), data["statement"]))
                    elif "question" in data:
                        texts.append((str(file_path), data["question"]))
```

### Example B: contrastive-decoding/model_comparison_helpers.py
```python
def load_jsonl_data(data_dir: str = "data/evals/anthropic-model-written-evals", ...):
    for jsonl_file in data_path.rglob("*.jsonl"):
        with jsonl_file.open() as f:
            for line in f:
                try:
                    data = json.loads(line)
                    keys = ','.join(sorted(data.keys()))
                    if keys in selected_keys:
                        for key in selected_keys[keys]:
                            if key in data:
                                prompts.append(data[key])
```

**Duplication**: Both load JSONL files but with different key selection strategies.

## 4. Model Device Handling

### Example A: soft_prompting/models/model_manager.py
```python
# Set up device-specific configurations
model_kwargs = {
    'device_map': None,  # Don't use device_map for MPS
    'torch_dtype': torch.float32,  # Use float32 for MPS
}

if self.device == 'cuda':
    model_kwargs.update({
        'device_map': 'auto',
        'torch_dtype': torch.float16,
        'load_in_8bit': getattr(self.config, 'load_in_8bit', False)
    })

# Explicitly move models to device for MPS or CPU
if self.device in ['mps', 'cpu']:
    model_1 = model_1.to(self.device)
    model_2 = model_2.to(self.device)
```

### Example B: contrastive-decoding/model_comparison_helpers.py
```python
# Determine the device to use
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

# Disable quantization for MPS
if device == "mps":
    no_quantize_starting_model = True
    bnb_config = None

# Set up device maps
if device == "mps" or device == "cpu":
    starting_model_device_map = device
    comparison_model_device_map = device
```

**Duplication**: Similar device detection and configuration logic implemented differently.

## 5. Clustering Implementation

### Example A: behavioural_clustering/evaluation/clustering.py
```python
class Clustering:
    def __init__(self, run_settings: RunSettings):
        self.algorithm_map = {
            "SpectralClustering": SpectralClustering,
            "KMeans": KMeans,
            "AgglomerativeClustering": AgglomerativeClustering,
            "OPTICS": OPTICS,
        }
    
    def _run_single_clustering(self, embeddings, clustering_algorithm, n_clusters, **kwargs):
        algorithm_class = self.algorithm_map[clustering_algorithm]
        if clustering_algorithm != "OPTICS":
            kwargs["n_clusters"] = n_clusters
        clustering = algorithm_class(**kwargs).fit(embeddings)
```

### Example B: interventions/auto_finetune_eval/auto_finetuning_interp.py
```python
# Direct sklearn usage without abstraction
kmeans_base = KMeans(n_clusters=n_clusters, random_state=0).fit(base_model_outputs_embeddings)
kmeans_finetuned = KMeans(n_clusters=n_clusters, random_state=0).fit(finetuned_model_outputs_embeddings)
```

**Duplication**: Clustering algorithms used directly vs through abstraction layer.

## 6. Cache Key Generation

### Example A: behavioural_clustering/utils/data_preparation.py
```python
def _generate_file_id(self, config: Dict[str, Any]) -> str:
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()
```

### Example B: behavioural_clustering/utils/embedding_manager.py
```python
def _get_embedding_key(self, statement: str, embedding_settings: EmbeddingSettings) -> str:
    return f"{statement}_{embedding_settings.embedding_model}"
```

**Duplication**: Different cache key strategies without unified approach.

## 7. Result Saving Patterns

### Example A: behavioural_clustering/utils/data_preparation.py
```python
def save_data(self, data: Any, data_type: str, config: Dict[str, Any]) -> str:
    file_id = self._generate_file_id(relevant_config)
    data_type_dir = self.data_dir / data_type
    data_type_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_type_dir / f"{file_id}.pkl"
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
```

### Example B: Repeated pattern in multiple modules with slight variations
```python
# Various pickle save operations without unified interface
pickle_dir = run_settings.directory_settings.pickle_dir
if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)
with open(pickle_dir / filename, "wb") as f:
    pickle.dump(data, f)
```

**Duplication**: File saving logic repeated with minor variations.