import os
import pandas as pd
from transformers import pipeline
from conllu import parse_incr


def pseudo_conll_to_conllu(path, output_path=None):
    """
    Convert pseudo conll to conllu format
    """
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("#"):
                line = line.replace(";", "\n#")
                line = line.replace("# ", "#")
                lines[i] = line

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(lines)


def conllu_parse_generator(path, file_index=False):
    with open(path, "r") as f:
        for tokenlist in parse_incr(f):
            yield tokenlist


def conllu_to_df(gen):
    """
    Process data from conllu generator to pandas dataframe

    params:
        gen: a conllu parse generator
    returns:
        df: a pandas dataframe
    """

    metadata = []
    for tokenlist in gen:
        sentence = []
        for token in tokenlist:
            sentence.append(token["form"])

        tokenlist.metadata["sentence"] = " ".join(sentence)
        metadata.append(tokenlist.metadata)

    df = pd.DataFrame(metadata)

    return df


pseudo_conll_to_conllu("de_filtered.txt", output_path="data/de_filtered_conllu.txt")
pseudo_conll_to_conllu("dem_filtered.txt", output_path="data/dem_filtered_conllu.txt")


conllu_gen = conllu_parse_generator("data/de_filtered_conllu.txt")
df_de = conllu_to_df(conllu_gen)

conllu_gen = conllu_parse_generator("data/dem_filtered_conllu.txt")
df_dem = conllu_to_df(conllu_gen)

df_de = df_de.rename(columns={"de-dem-dom_id": "token_id"})
df_dem = df_dem.rename(columns={"de-dem-dom_id": "token_id"})


pipe = pipeline("ner", "../deformer/deformer_v2", tokenizer="../deformer/deformer_v2", device=0)

df_de["sentence_lower"] = df_de["sentence"].str.lower()
df_dem["sentence_lower"] = df_dem["sentence"].str.lower()
df_de["preds"] = df_de["sentence_lower"].apply(lambda x: pipe(x))
df_dem["preds"] = df_dem["sentence_lower"].apply(lambda x: pipe(x))

# split sentence into words and count the start and end character index of the word in index token_id
df_de["word_list"] = df_de["sentence"].str.split()
df_dem["word_list"] = df_dem["sentence"].str.split()


def get_char_index(row):
    start = 0
    end = len(row["word_list"][0])

    if int(row["token_id"]) == 1:
        return start, end

    for i, word in enumerate(row["word_list"][1:]):
        start += end - start + 1
        end += len(word) + 1

        if (i + 2) == int(row["token_id"]):
            return start, end


df_de["char_index"] = df_de.apply(lambda row: get_char_index(row), axis=1)
df_de[["start", "end"]] = pd.DataFrame(df_de["char_index"].tolist(), index=df_de.index)
df_dem["char_index"] = df_dem.apply(lambda row: get_char_index(row), axis=1)
df_dem[["start", "end"]] = pd.DataFrame(df_dem["char_index"].tolist(), index=df_dem.index)


# Loop over df_de["preds"] and extract the index where df_de["start"] == df_de["preds"][i]["start"]
def get_pred_index(row):
    for i, pred in enumerate(row["preds"]):
        if row["start"] == pred["start"]:
            return pred


df_de["predicted_label"] = df_de.apply(lambda row: get_pred_index(row), axis=1)
df_dem["predicted_label"] = df_dem.apply(lambda row: get_pred_index(row), axis=1)

# Spread dictionary predicted_label column over multiple columns
df_de[["prediction", "score", "index", "word", "start", "end"]] = pd.DataFrame(
    df_de["predicted_label"].tolist(), index=df_de.index
)
df_de[["sent_id", "token_id", "sentence", "prediction", "score", "index", "word", "start", "end"]][50:100].values

df_dem[["prediction", "score", "index", "word", "start", "end"]] = pd.DataFrame(
    df_dem["predicted_label"].tolist(), index=df_dem.index
)
df_dem[["sent_id", "token_id", "sentence", "prediction", "score", "index", "word", "start", "end"]][50:100].values


# Use regex to remove spaces between alphanumerics and punctuation in sentence_lower
import re

# Save to tsv
df_de[["sent_id", "token_id", "prediction", "score"]].to_csv("data/de_filtered_preds.tsv", sep="\t", index=False)
df_dem[["sent_id", "token_id", "prediction", "score"]].to_csv("data/dem_filtered_preds.tsv", sep="\t", index=False)
