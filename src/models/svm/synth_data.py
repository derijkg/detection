import random
import zlib
import numpy as np
import pandas as pd


def mix_abstract(human_sentences, llm_sentences, target_ratio, random_state=42):
  """Substitutes target_ratio % of human sentences with corresponding LLM sentences.

  Assumes 1-to-1 parallel sentence lists.
  """
  n_sentences = len(human_sentences)
  if n_sentences != len(llm_sentences):
    raise ValueError("Human and LLM sentence lists must be equal in length.")

  # Determine exact number of sentences to replace
  k = int(round(target_ratio * n_sentences))

  # Clamp k between 0 and n_sentences
  k = max(0, min(n_sentences, k))

  # Set seed for reproducibility
  rng = random.Random(random_state)

  # Randomly select indices for LLM replacement
  llm_indices = set(rng.sample(range(n_sentences), k))

  # Construct the mixed list of sentences
  mixed_sentences = [
      llm_sentences[i] if i in llm_indices else human_sentences[i]
      for i in range(n_sentences)
  ]

  actual_ratio = k / n_sentences if n_sentences > 0 else 0.0
  mixed_text = " ".join(mixed_sentences)

  return mixed_text, actual_ratio, sorted(list(llm_indices))


def generate_mixed_test_dataset(
    test_df, target_ratios=[0.25, 0.50, 0.75], seed=42
):
  """test_df expected columns:

  - 'doc_id': Unique abstract identifier
  - 'human_sentences': List[str]
  - 'llm_sentences': List[str]
  """
  mixed_records = []

  for idx, row in test_df.iterrows():
    doc_id = row['doc_id']
    h_sents = row['human_sentences']
    l_sents = row['llm_sentences']

    # Skip very short abstracts where percentages cannot be meaningfully split
    if len(h_sents) < 4 or len(h_sents) != len(l_sents):
      continue

    for ratio in target_ratios:
      # Generate a unique, deterministic seed per document and ratio using CRC32
      seed_str = f"{seed}_{doc_id}_{ratio}"
      pair_seed = zlib.crc32(seed_str.encode("utf-8"))

      mixed_text, actual_ratio, llm_indices = mix_abstract(
          h_sents, l_sents, target_ratio=ratio, random_state=pair_seed
      )

      mixed_records.append({
          "doc_id": doc_id,
          "target_ratio": ratio,
          "actual_ratio": actual_ratio,
          "llm_sentence_indices": llm_indices,
          "mixed_text": mixed_text,
          "num_sentences": len(h_sents),
      })

  return pd.DataFrame(mixed_records)