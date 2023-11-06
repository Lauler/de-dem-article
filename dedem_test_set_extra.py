import re
import pandas as pd
from transformers import pipeline, BertForTokenClassification, AutoTokenizer
import torch
from tqdm import tqdm
from src.dataset import DedemDataset, custom_collate_fn

tqdm.pandas()

model = BertForTokenClassification.from_pretrained("Lauler/deformer").eval().cuda()

# pipe = pipeline("ner", "Lauler/DeFormer", tokenizer="Lauler/DeFormer", device=0)
df = pd.read_csv("data/dem_test_set.tsv", sep="\t")


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


df["sentence"] = df["sentence"].progress_apply(lambda x: replace_patterns(x))


def find_token_char(df):
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
    df = df.drop(columns=["sentence_split"])

    return df


df = find_token_char(df)

inputs = df["sentence"].tolist()

# Create dataset class
dataset = DedemDataset(inputs)

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn, num_workers=4
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

df = df[(df["correct"] == "dem") & (df["actual_written"] == "de")]

df = df.drop(columns=["preds", "labels", "scores", "preds_only", "labels_only", "token_index"])
df["prediction"] = df["preds_text"].str.lower()
df = df.rename(columns={"scores_only": "score"})
df = df[
    ["token_id", "begin_char", "end_char", "correct", "actual_written", "prediction", "score", "sentence"]
].reset_index(drop=True)
df.to_csv("data/results/de_som_ska_var_dem.tsv", sep="\t", index=False)
