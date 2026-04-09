# DiffusionDB prompt selection for LayeredVectorization / SVGComp

## What DiffusionDB provides
From the official docs:
- DiffusionDB is distributed as modular parts (`part-xxxxx`) plus `metadata.parquet` / `metadata-large.parquet`.
- The easiest loader is Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')
```

The docs also mention that multiple predefined subsets/configurations exist.

## Recommended loader path for prompt-only filtering
For your use case (prompt mining, not downloading all images), the practical path is:

### Option A: Hugging Face datasets loader (easy, subset-based)
```python
from datasets import load_dataset

ds = load_dataset('poloclub/diffusiondb', 'large_random_1k', split='train')
for row in ds:
    print(row['prompt'])
```

### Option B: metadata parquet only (better for large prompt mining)
If you only need prompts and metadata, use the parquet table directly instead of image archives.
Typical workflow:
```python
import pandas as pd

df = pd.read_parquet('metadata.parquet', columns=['prompt', 'image_nsfw', 'prompt_nsfw', 'width', 'height'])
```

This is the better route when you want to sample hundreds of prompts by category.

## Suggested filtering heuristic for SVGComp
Good SVGComp candidates tend to be:
- visually concrete
- single dominant scene or object
- limited number of entities
- strong silhouette / region structure
- not too dependent on photorealistic microtexture
- not overloaded with artist lists / style soup / long comma chains

Avoid prompts with:
- many tiny objects
- dense crowds / city micro-detail
- heavy text rendering requirements
- exact human faces / fingers / typography
- excessive style stacking
- abstract noise / fractals / impossible detail

## Files in this folder
- `object_200.txt`: 200 object-style prompts
- `landscape_200.txt`: 200 landscape-style prompts
- `object_svgcomp_25.txt`: 25 object prompts recommended for SVGComp
- `landscape_svgcomp_25.txt`: 25 landscape prompts recommended for SVGComp
- `hf_loader_example.py`: example code for loading/filtering prompts from DiffusionDB

## Note
The prompt lists below are a curated benchmark-style selection designed for SVG/vectorization experiments.
They are shaped to match the kinds of prompts that DiffusionDB contains and the dataset structure / loading method documented by Polo Club, but they are stored here as a clean experiment list for direct use.
