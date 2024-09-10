# pyright: basic

import math


def calculate_perplexity(f: str) -> float:
    total_log_prob = 0.0
    total_tokens = 0

    with open(f, "r") as file:
        # Read token IDs
        token_ids = list(map(int, file.readline().strip().split(": ")[1].split()))

        # Process probability distributions
        for line, next_token in zip(file, token_ids[1:]):  # Start from the second token
            probs = list(map(float, line.strip().split()))
            if next_token < len(probs):
                prob = probs[next_token]
                if prob > 0:
                    total_log_prob += math.log(prob)
                    total_tokens += 1

    average_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(-average_log_prob) * 100

    return perplexity


p = calculate_perplexity("perplexity_data.txt")
print(f"Perplexity: {p}")

from collections import defaultdict


def train_char_bigram_model(text):
    # Count bigram occurrences
    bigram_counts = defaultdict(lambda: defaultdict(int))
    char_counts = defaultdict(int)

    for i in range(len(text) - 1):
        char, next_char = text[i], text[i + 1]
        bigram_counts[char][next_char] += 1
        char_counts[char] += 1

    # Calculate probabilities
    bigram_probs = defaultdict(lambda: defaultdict(float))
    for char, next_chars in bigram_counts.items():
        for next_char, count in next_chars.items():
            bigram_probs[char][next_char] = count / char_counts[char]

    return bigram_probs


def calculate_perplexity_ngram(text, model):
    log_prob = 0
    n = len(text) - 1  # number of bigrams

    for i in range(n):
        char, next_char = text[i], text[i + 1]
        prob = model[char][next_char]
        if prob == 0:
            prob = 1e-10  # smoothing to avoid log(0)
        log_prob += math.log2(prob)

    perplexity = 2 ** (-log_prob / n)
    return perplexity


# Example usage
training_text = (
    "The quick brown fox jumps over the lazy dog. The lazy dog sleeps all day."
)
test_text = "The quick brown fox jumps over the fence."

# Train the model
bigram_model = train_char_bigram_model(training_text)

# Calculate perplexity
perplexity = calculate_perplexity_ngram(test_text, bigram_model)
print(f"N-gram Perplexity: {perplexity}")
