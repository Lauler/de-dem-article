import torch
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import BertForTokenClassification, AutoTokenizer
from tqdm import tqdm
from src.dataset import DedemDataset, custom_collate_fn
from src.preprocessing import find_token_char, replace_patterns, character_span_to_token_mapping, is_relative_clause

tqdm.pandas()


df = pd.concat(
    [
        pd.read_parquet("data/reddit/2010-04-08_2012-04-08.parquet"),
        pd.read_parquet("data/reddit/2012-04-08_2014-04-08.parquet"),
        pd.read_parquet("data/reddit/2014-04-08_2018-04-08.parquet"),
        pd.read_parquet("data/reddit/2018-04-08_2022-04-08.parquet"),
        pd.read_parquet("data/reddit/2022-04-08_2023-04-30.parquet"),
    ]
)

df.reset_index(drop=True).to_parquet("data/reddit/2010-04-08_2023-04-30_dedem_reddit.parquet", index=False)


def filter_dedem_sentences(sentences):
    """
    Keep only sentences with de/dem.
    """

    dedem_pattern = "(?<!\w)[Dd][Ee][Mm]?(?![\w\*])"
    dedem_sentences = filter(lambda sentence: bool(re.search(dedem_pattern, sentence)), sentences)

    # Don't match single word sentences
    dedem_sentences = filter(lambda sentence: len(sentence.split()) > 1, dedem_sentences)

    return list(dedem_sentences)


def remove_emoji(text):
    # https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

    emoji_pattern = re.compile("[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)

    return emoji_pattern.sub("", text)


def preprocess_comments(df):
    # Remove sentences that are quotes from other comments.
    # Comments that start with ">" and end with "\n\n" are quotes.
    df["body_temp"] = df["body"].str.replace(">.*\n\n", "", regex=True)
    df["body_temp"] = df["body_temp"].str.replace("\n\n\n", "\n\n", regex=True)

    # Split comment body into list of sentences
    df["sentences"] = df["body_temp"].apply(lambda doc: sent_tokenize(doc, language="swedish"))
    df = df.drop(columns=["body_temp"])

    # Keep only sentences with de/dem
    df["sentences"] = df["sentences"].apply(lambda sen: filter_dedem_sentences(sen))
    df["sentences_filter"] = df["sentences"].apply(lambda x: any([len(y) != 0 for y in x]))

    # Split sentences also on new paragraphs "\n\n" (in case someone doesn't use punctuation)
    df["sentences"] = df["sentences"].apply(lambda sens: [sen.splitlines() for sen in sens])
    # Flatten list of lists and remove empty sentences consisting of only ''.
    df["sentences"] = df["sentences"].apply(lambda sens: [sen for split_sens in sens for sen in split_sens])
    df["sentences"] = [[sen for sen in sens if len(sen) > 0] for sens in df["sentences"]]

    # Remove emojis
    df["sentences"] = df["sentences"].apply(lambda sens: [remove_emoji(sen) for sen in sens])

    # Strip whitespace before and after sentence.
    df["sentences"] = df["sentences"].apply(lambda sens: [sen.strip() for sen in sens])

    # Remove 2 or more spaces in a row and replace by single space.
    df["sentences"] = df["sentences"].apply(lambda sens: [re.sub(" {2,}", " ", sen) for sen in sens])

    return df


df = preprocess_comments(df)

# Removes mostly comments where "de" or "dem" are only used in quotes (quoting other comments).
# Also removes deleted and unavailable comments.
df = df[df["sentences_filter"] == True].reset_index(drop=True)

df = df.drop(columns="body")
df = df.explode("sentences")

# Find all character index spans of "de" and "dem" in each sentence using re.finditer
df["de_spans"] = df["sentences"].apply(
    lambda sen: [m.span() for m in re.finditer("(?<!\w)[Dd][Ee][Mm]?(?![\w\*])", sen)]
)
df = df.explode("de_spans")
df = df[~df["de_spans"].isna()].reset_index(drop=True)
df["begin_char"] = df["de_spans"].apply(lambda span: span[0])
df["end_char"] = df["de_spans"].apply(lambda span: span[1])

df = df.drop(columns=["de_spans", "sentences_filter"])
df.to_parquet("data/dem_filtered_reddit.parquet", index=False)


#### Model

model = BertForTokenClassification.from_pretrained("Lauler/deformer").eval().cuda()

df = pd.read_parquet("data/dem_filtered_reddit.parquet")
df = df.rename(columns={"sentences": "sentence"})

df["sentence"] = df["sentence"].str.normalize("NFC")
# Remove whitespace \t \r \n etc
df["sentence"] = df["sentence"].str.replace(r"\s", " ", regex=True)
# Remove multiple spaces
df["sentence"] = df["sentence"].str.replace(r"\s+", " ", regex=True)

df["sentence_length"] = df["sentence"].str.len()
df = df[df["sentence_length"] <= 20000].reset_index(drop=True)  # Max characters


df["sentence"] = df["sentence"].progress_apply(lambda x: replace_patterns(x))

inputs = df["sentence"].tolist()

# Create dataset class
dataset = DedemDataset(inputs)

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, num_workers=4
)

# Run model
preds = []
labels = []
scores = []
for batch in tqdm(dataloader):
    with torch.inference_mode(), torch.cuda.amp.autocast():
        batch_inputs = {k: v.cuda() for k, v in batch.items()}
        outputs = model(batch_inputs["input_ids"], batch_inputs["attention_mask"])
        predictions = outputs["logits"].argmax(dim=2)
        # Extract the highest score for each token
        scores_temp = torch.softmax(outputs["logits"], dim=2).max(dim=2).values

        # Boolean indexing to get only the predictions/labels/scores for the actual tokens (not padding tokens)
        predictions = [pred[mask == 1] for pred, mask in zip(predictions.cpu(), batch["attention_mask"].cpu())]
        labels_temp = [label[mask == 1] for label, mask in zip(batch["labels"].cpu(), batch["attention_mask"].cpu())]
        scores_temp = [score[mask == 1] for score, mask in zip(scores_temp.cpu(), batch["attention_mask"].cpu())]
        preds.extend(predictions)
        labels.extend(labels_temp)
        scores.extend(scores_temp)


preds = [p.tolist() for p in preds]
labels = [l.tolist() for l in labels]
scores = [s.tolist() for s in scores]
df["preds"] = preds
df["labels"] = labels
df["scores"] = scores


tokenizer = AutoTokenizer.from_pretrained("Lauler/deformer")
df["token_index"] = df[["sentence", "begin_char", "end_char"]].progress_apply(
    lambda x: character_span_to_token_mapping(tokenizer(x.sentence), x.begin_char, x.end_char), axis=1
)

# Some token_id are not correct, we remove them
df = df[df["token_index"].apply(lambda x: len(x)) == 1].reset_index(drop=True)
df["token_index"] = df["token_index"].str[0]

# We remove the sentences that are too long (outside of models maximum sequence length 512)
df = df[df["token_index"] < 512].reset_index(drop=True)

# Select token_index from preds, labels and scores
df["preds_only"] = df.apply(lambda x: x["preds"][x["token_index"]], axis=1)
df["labels_only"] = df.apply(lambda x: x["labels"][x["token_index"]], axis=1)
df["scores_only"] = df.apply(lambda x: x["scores"][x["token_index"]], axis=1)

df["labels_text"] = df["labels_only"].map(model.config.id2label)
df["preds_text"] = df["preds_only"].map(model.config.id2label)

pd.crosstab(df["labels_text"], df["preds_text"])


# We consider only predictions that are DE, DEM or DET
df = df[df["preds_text"].isin(["DE", "DEM", "DET"])].reset_index(drop=True)

# Remove all "ord" labels (de, dem, det, enda, ända become "ord" if they are the prefix of a longer word)
# E.g. "detdär" becomes "ord" instead of "det" label because "det" is a prefix to ##där .
df = df[df["labels_text"] != "ord"]
df = df.drop(columns=["preds", "labels", "scores"])

df_dedem = df[df["labels_text"].isin(["DE", "DEM"])].reset_index(drop=True)

df_dedem["maincorpus"] = "reddit"

# Convert epoch to date and time using pd.to_datetime
df_dedem["date"] = df_dedem["created"].progress_apply(lambda x: pd.to_datetime(x, unit="s"))

# Extract the year
df_dedem["year"] = df_dedem["date"].dt.year


df_dedem = df_dedem[
    [
        "id",
        "link_id",
        "permalink",
        "score",
        "author",
        "year",
        "date",
        "created",
        "maincorpus",
        "sentence",
        "begin_char",
        "end_char",
        "sentence_length",
        "labels_text",
        "preds_text",
        "scores_only",
        "token_index",
    ]
]

df_dedem = pd.read_parquet("data/results/reddit_results.parquet")
df_dedem["sentence_split"] = df_dedem["sentence"].str.split(" ")

# Find token index for "de" or "dem" in sentence_split and add it to column token_id

df_dedem["token_id"] = -100  # Filler value


def is_relative_clause(row):
    """
    Check if "som" appears after end_char in sentence.
    "som" should not be part of a word, e.g. "sommar" or "somliga".
    """
    if re.search(
        r"(?<!\w)[Dd][Ee][Mm]? som[\.,;\"]?(?![\w])",
        row["sentence"][(row["begin_char"] + 0) : (row["end_char"] + 5)],
    ):
        return True
    else:
        return False


df_dedem["is_relative_clause"] = df_dedem.apply(lambda x: is_relative_clause(x), axis=1)
df_dedem = df_dedem.drop(columns=["sentence_split"])

# Export as parquet
df_dedem.to_parquet("data/results/reddit_results.parquet", index=False)
