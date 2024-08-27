import pandas as pd
import math
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import string
from itertools import groupby
import numpy as np

def load_data(file_path, delimiter='\t'):
    """Load data from a CSV file."""
    print("Loading data from CSV file...")
    df = pd.read_csv(file_path, delimiter=delimiter)
    print("Data loaded successfully.")
    return df

def process_word_column(df):
    """Ensure all entries in the 'word' column are strings and handle NaN values."""
    print("Processing 'word' column to ensure all entries are strings and handle NaN values...")
    df['word'] = df['word'].astype(str).fillna('')
    print("'Word' column processed.")
    return df

def calculate_entropy(word):
    if not word:
        return 0
    counts = Counter(word)
    probabilities = [count / len(word) for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

def count_digits(word):
    return sum(c.isdigit() for c in word)

def count_uppercase(word):
    return sum(c.isupper() for c in word)

def count_special_chars(word):
    special_characters = string.punctuation
    return sum(c in special_characters for c in word)

def count_vowels(word):
    vowels = "aeiouAEIOU"
    return sum(c in vowels for c in word)

def count_consonants(word):
    vowels = "aeiouAEIOU"
    return sum(c.isalpha() and c not in vowels for c in word)

def count_repeated_chars(word):
    return sum(1 for i in range(1, len(word)) if word[i] == word[i - 1])

def count_unique_chars(word):
    return len(set(word))

def max_consecutive_repeats(word):
    return max((sum(1 for _ in group) for _, group in groupby(word)), default=1)

def has_special_chars(word):
    special_characters = string.punctuation
    return int(any(c in special_characters for c in word))

def has_digits(word):
    return int(any(c.isdigit() for c in word))

def has_uppercase(word):
    return int(any(c.isupper() for c in word))

def average_unicode_value(word):
    if not word:
        return 0
    return sum(ord(c) for c in word) / len(word)

def ratio_special_chars_to_length(word):
    length = len(word)
    if length == 0:
        return 0
    special_characters = string.punctuation
    special_char_count = sum(c in special_characters for c in word)
    return special_char_count / length

def ratio_digits_to_length(word):
    length = len(word)
    if length == 0:
        return 0
    digit_count = sum(c.isdigit() for c in word)
    return digit_count / length

def ratio_vowels_to_length(word):
    length = len(word)
    if length == 0:
        return 0
    vowels = 'aeiouAEIOU'
    vowel_count = sum(c in vowels for c in word)
    return vowel_count / length

def ratio_consonants_to_length(word):
    length = len(word)
    if length == 0:
        return 0
    consonants = ''.join(set(string.ascii_letters) - set('aeiouAEIOU'))
    consonant_count = sum(c in consonants for c in word)
    return consonant_count / length

def word_symmetry(word):
    if len(word) <= 1:
        return 0
    return sum(1 for i in range(len(word) // 2) if word[i] == word[-(i + 1)]) / (len(word) // 2)

def is_palindrome(word):
    return int(word == word[::-1])

def ratio_unique_chars_to_length(word):
    length = len(word)
    if length == 0:
        return 0
    unique_char_count = len(set(word))
    return unique_char_count / length

def longest_alphabetical_sequence(word):
    if len(word) == 0:
        return 0
    max_len = 1
    curr_len = 1
    for i in range(1, len(word)):
        if word[i] >= word[i - 1]:
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 1
    return max_len

def proportion_uppercase(word):
    length = len(word)
    if length == 0:
        return 0
    uppercase_count = sum(c.isupper() for c in word)
    return uppercase_count / length

def count_white_spaces(word):
    return sum(c.isspace() for c in word)

def extract_features(word):
    """Extract features from a single word."""
    features = {
        # 'length': len(word), # This feature is not included in the final model du to high correlation with other features
        'digit_count': count_digits(word),
        'uppercase_count': count_uppercase(word),
        'special_char_count': count_special_chars(word),
        'entropy': calculate_entropy(word),
        'vowel_count': count_vowels(word),
        'consonant_count': count_consonants(word),
        'repeated_char_count': count_repeated_chars(word),
        'unique_char_count': count_unique_chars(word),
        'max_consecutive_repeats': max_consecutive_repeats(word),
        'has_special_chars': has_special_chars(word),
        'has_digits': has_digits(word),
        'has_uppercase': has_uppercase(word),
        'average_unicode': average_unicode_value(word),
        'special_char_ratio': ratio_special_chars_to_length(word),
        'digit_ratio': ratio_digits_to_length(word),
        'vowel_ratio': ratio_vowels_to_length(word),
        'consonant_ratio': ratio_consonants_to_length(word),
        'symmetry': word_symmetry(word),
        # 'is_palindrome': is_palindrome(word), # This feature is not included in the final model du not being useful
        'unique_char_ratio': ratio_unique_chars_to_length(word),
        'longest_alpha_seq': longest_alphabetical_sequence(word),
        'uppercase_proportion': proportion_uppercase(word),
        'white_space_count': count_white_spaces(word)
    }
    return features