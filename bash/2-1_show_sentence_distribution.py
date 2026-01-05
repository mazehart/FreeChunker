from datasets import load_from_disk
import matplotlib.pyplot as plt
from collections import Counter
import json
import numpy as np

# Dataset path
DATASET_PATH = "dataset/wikitext_train"
dataset = load_from_disk(DATASET_PATH)

# Count number of sentences per sample
sentence_counts = [len(item['sentences']) for item in dataset]

# Plot histogram with improved style
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(sentence_counts, bins=30, edgecolor='black', color='#4C72B0', alpha=0.85)

# Labels and title
ax.set_xlabel('Number of Sentences', fontsize=14)
ax.set_ylabel('Number of Samples', fontsize=14)
ax.set_title('Distribution of Sentence Counts per Sample', fontsize=16, weight='bold')

# Add mean and median lines
mean_val = np.mean(sentence_counts)
median_val = np.median(sentence_counts)
ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}')
ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.1f}')

# Add legend
ax.legend(fontsize=12)

# Add grid
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('sentence_distribution.png', dpi=150)
plt.show()

# Save distribution data to json
counter = Counter(sentence_counts)
with open('sentence_count_distribution.json', 'w') as f:
    json.dump(counter, f, ensure_ascii=False, indent=2) 