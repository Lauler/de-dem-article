import csv
import torch
import pandas as pd
from transformers import BertForTokenClassification, AutoTokenizer
from tqdm import tqdm
from src.dataset import DedemDataset, custom_collate_fn
from src.preprocessing import (
    find_token_char,
    replace_patterns,
    character_span_to_token_mapping,
    is_relative_clause,
)

tqdm.pandas()
model = BertForTokenClassification.from_pretrained("Lauler/deformer").eval().cuda()

df = pd.concat(
    [
        pd.read_csv(
            "data/pp_demsom_desom_gp.tsv",
            sep="\t",
            engine="python",
            quoting=csv.QUOTE_NONE,
        ),
        pd.read_csv(
            "data/pp_demsom_desom_svt.tsv",
            sep="\t",
            engine="python",
            quoting=csv.QUOTE_NONE,
        ),
        pd.read_csv(
            "data/pp_demsom_desom_bloggmix.tsv",
            sep="\t",
            engine="python",
            quoting=csv.QUOTE_NONE,
        ),
        pd.read_csv(
            "data/pp_de(m)_som_familjeliv_age.tsv",
            sep="\t",
            engine="python",
            quoting=csv.QUOTE_NONE,
        ),
    ]
)

# Filter out yob == 1970 (We don't do this for corpus level results)
# df = df[df["yob"] != 1970].reset_index(drop=True)

df["sentence"] = df["sentence"].str.normalize("NFC")
# Remove whitespace \t \r \n etc
df["sentence"] = df["sentence"].str.replace(r"\s", " ", regex=True)
# Remove multiple spaces
df["sentence"] = df["sentence"].str.replace(r"\s+", " ", regex=True)

df["sentence_length"] = df["sentence"].str.len()
df = df[df["sentence_length"] <= 20000].reset_index(drop=True)  # Max characters
df["token_id"] = df["token_id"] - 1  # Convert token_id to 0-indexed

df = find_token_char(df)

# Lowercase instances of de/dem/det since the model is trained on lowercased instances
df["sentence"] = df["sentence"].progress_apply(lambda x: replace_patterns(x))

inputs = df["sentence"].tolist()


# Create dataset class
dataset = DedemDataset(inputs)

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn, num_workers=4
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
        predictions = [
            pred[mask == 1]
            for pred, mask in zip(predictions.cpu(), batch["attention_mask"].cpu())
        ]
        labels_temp = [
            label[mask == 1]
            for label, mask in zip(batch["labels"].cpu(), batch["attention_mask"].cpu())
        ]
        scores_temp = [
            score[mask == 1]
            for score, mask in zip(scores_temp.cpu(), batch["attention_mask"].cpu())
        ]
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
    lambda x: character_span_to_token_mapping(
        tokenizer(x.sentence), x.begin_char, x.end_char
    ),
    axis=1,
)

# Some token_id are not correct, we remove them by keeping only valid predictions
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

# Export as parquet
df_dedem.to_parquet("data/results/pp_demsom_results.parquet", index=False)
