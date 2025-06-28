import random
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

SPIECE_WHITESPACE = "▁"
GPT_BPE_WHITESPACE = "Ġ"


def get_new_phrase_tokenized_ids(new_phrase, tokenizer: PreTrainedTokenizer, tokenizer_path=""):
    """
    Once your stare into the void for too long, the void stares back at you.
    In this case, the void is the `tokenizers` implementation *devoid* of proper handling of prefix whitespaces.

    It is very difficiult to reliably tokenize a span of text that doesn't start with a whitespace.
    `add_prefix_space=False` is supposed to handle this, but it doesn't work as expected in my tests.
    We use the following strategy to reliably split a span of text (e.g. `new_phrase`) into tokens and correctly handle the prefix whitespace:
    (1) We add the BOS token to the beginning of the text.
    (2) We tokenize the text with the tokenizer.
    (3) We remove the BOS token id from the output.
    """
    # yes, it's even difficult to get the actual whitespace token
    whitespace_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(f"{tokenizer.bos_token} ", add_special_tokens=False))[1]
    new_phrase_starts_with_whitespace = new_phrase.startswith(" ") or new_phrase.startswith(whitespace_token)

    new_phrase_w_bos = f"{tokenizer.bos_token}{new_phrase}"
    new_phrase_tokenized_ids = tokenizer.encode(new_phrase_w_bos, return_tensors="pt", add_special_tokens=False)[0][1:]

    # sanity checks
    assert new_phrase_tokenized_ids[0] != tokenizer.bos_token_id
    if new_phrase_starts_with_whitespace:
        assert tokenizer.convert_ids_to_tokens(new_phrase_tokenized_ids[0].item()).startswith(whitespace_token)
    else:
        assert not tokenizer.convert_ids_to_tokens(new_phrase_tokenized_ids[0].item()).startswith(whitespace_token)

    return new_phrase_tokenized_ids


def generate_samples_with_patterns(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokens_to_pattern: dict[str, Iterable[int]],
    num_samples_per_pattern: int,
    seed=42,
    max_length=50,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    max_length = max_length + 1  # +1 for the BOS token

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer does not have a BOS token. Please use a tokenizer that supports BOS tokens.")

    token_to_samples = {}

    for token, pattern in tqdm(tokens_to_pattern.items(), desc="Generating samples", total=len(tokens_to_pattern)):
        pattern_with_bos = torch.tensor([tokenizer.bos_token_id] + list(pattern), device=model.device).unsqueeze(0)

        token_to_samples[token] = model.generate(
            pattern_with_bos,
            do_sample=True,
            max_length=max_length,
            min_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            num_return_sequences=num_samples_per_pattern,  # We're already batching
        ).to("cpu")
        sample_sequence = token_to_samples[token][0]
        print(f"\nToken '{token}' (pattern: {tokenizer.decode(pattern, skip_special_tokens=True)}) - sample sequence:")
        print(f"Generated text: {tokenizer.decode(sample_sequence, skip_special_tokens=True)}\n-----------\n\n")
    return token_to_samples
