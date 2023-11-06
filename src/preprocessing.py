import pandas as pd
import re


def find_token_char(df):
    """
    Find the begin/end character index for token_id in the sentence
    (token_id is the index of the token we're interested in).
    """

    # Split sentence on space and count the character index of each token
    df["sentence_split"] = df["sentence"].str.split(" ")
    # Beginning and end character index for each token
    df["begin_char"] = df["sentence_split"].progress_apply(
        lambda x: [0] + [len(" ".join(x[:i])) + 1 for i in range(1, len(x) + 1)]
    )
    df["end_char"] = df["sentence_split"].progress_apply(
        lambda x: [len(" ".join(x[:i])) for i in range(1, len(x) + 1)]
    )

    # Choose begin_char from token_id
    df["begin_char"] = df.apply(lambda x: x["begin_char"][x["token_id"]], axis=1)
    # Choose end_char from token_id
    df["end_char"] = df.apply(lambda x: x["end_char"][x["token_id"]], axis=1)

    # Check if the token is followed by a "som"
    df["is_relative_clause"] = df.apply(is_relative_clause, axis=1)

    df = df.drop(columns=["sentence_split"])

    return df


def replace_patterns(sentence):
    """
    Replace different cases of words with lower case letters
    (We trained the model on lower case letters)
    """
    de_pattern = "(?<!\w)[D][Ee](?!\w)"
    dem_pattern = "(?<!\w)[D][Ee][Mm](?!\w)"
    det_pattern = "(?<!\w)[D][Ee][Tt](?!\w)"
    enda_pattern = "(?<!\w)[Ee][Nn][Dd][Aa](?!\w)"
    anda_pattern = "(?<!\w)[Ää][Nn][Dd][Aa](?!\w)"

    sentence = re.sub(de_pattern, "de", sentence)
    sentence = re.sub(dem_pattern, "dem", sentence)
    sentence = re.sub(det_pattern, "det", sentence)
    sentence = re.sub(enda_pattern, "enda", sentence)
    sentence = re.sub(anda_pattern, "ända", sentence)
    return sentence


def character_span_to_token_mapping(tokenized_text, start_char, end_char):
    """Maps character positions to token positions.

    Args:
        tokenized_text (transformers.tokenization_utils_base.BatchEncoding): The tokenized text.
        start_char (int): The start character position.
        end_char (int): The end character position.

    Returns:
        A set of token indices corresponding to the character spans overlapped by the given
        character span in the original text.
    """
    tokens = []

    for char_index in range(start_char, end_char):
        token_id = tokenized_text.char_to_token(char_index)
        if token_id is not None:
            tokens.append(token_id)
        else:
            ValueError("Character position out of bounds")

    return sorted(set(tokens))


def is_relative_clause(row):
    """
    Return True if "som" is the word after de/dem
    (the word after de/dem is in position token_id + 1)
    """
    try:
        if (row["sentence_split"][row["token_id"] + 1] == "som") | (
            row["sentence_split"][row["token_id"] + 1] == "SOM"
        ):
            return True
        else:
            return False
    except IndexError:
        # If de/dem is the last word in the sentence
        return False
