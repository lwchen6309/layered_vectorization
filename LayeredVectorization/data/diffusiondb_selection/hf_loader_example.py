from datasets import load_dataset

# Example: load a predefined subset from Hugging Face
# Docs example uses: large_random_1k

ds = load_dataset('poloclub/diffusiondb', 'large_random_1k', split='train')

print(ds)
print(ds.column_names)
print(ds[0]['prompt'])

# Example heuristic filtering for prompt mining
OBJECT_WORDS = [
    'vase', 'chair', 'lamp', 'teapot', 'bottle', 'helmet', 'watch', 'camera', 'robot',
    'backpack', 'shoe', 'mug', 'car', 'bicycle', 'violin', 'guitar', 'statue', 'mask'
]
LANDSCAPE_WORDS = [
    'landscape', 'mountain', 'forest', 'lake', 'river', 'desert', 'coast', 'beach',
    'valley', 'waterfall', 'meadow', 'island', 'sunset', 'cliff', 'snow'
]

def simple_match(prompt, vocab):
    p = prompt.lower()
    return any(w in p for w in vocab)

# Pull prompt-only views
object_prompts = []
landscape_prompts = []

for row in ds:
    prompt = row['prompt']
    if row.get('prompt_nsfw', 0) and row['prompt_nsfw'] > 0.15:
        continue
    if simple_match(prompt, OBJECT_WORDS):
        object_prompts.append(prompt)
    if simple_match(prompt, LANDSCAPE_WORDS):
        landscape_prompts.append(prompt)

print('object candidates:', len(object_prompts))
print('landscape candidates:', len(landscape_prompts))
print('\nSample object prompts:')
for p in object_prompts[:20]:
    print('-', p)
print('\nSample landscape prompts:')
for p in landscape_prompts[:20]:
    print('-', p)
