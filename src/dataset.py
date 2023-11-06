import torch
from transformers import AutoTokenizer


class DedemDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        self.tokenizer = AutoTokenizer.from_pretrained("Lauler/deformer")

        self.de_id = self.tokenizer.convert_tokens_to_ids("de")
        self.De_id = self.tokenizer.convert_tokens_to_ids("De")
        self.dem_id = self.tokenizer.convert_tokens_to_ids("dem")
        self.Dem_id = self.tokenizer.convert_tokens_to_ids("Dem")
        self.det_id = self.tokenizer.convert_tokens_to_ids("det")
        self.Det_id = self.tokenizer.convert_tokens_to_ids("Det")
        self.enda_id = self.tokenizer.convert_tokens_to_ids("enda")
        self.Enda_id = self.tokenizer.convert_tokens_to_ids("Enda")
        self.anda_id = self.tokenizer.convert_tokens_to_ids("ända")
        self.Anda_id = self.tokenizer.convert_tokens_to_ids("Ända")

        self.de_label = 1
        self.dem_label = 2
        self.det_label = 3
        self.enda_label = 4
        self.anda_label = 5

    def __len__(self):
        return len(self.inputs)

    def get_written_word(self, tokens):
        """
        Construct vector of labels
        (the word that is written in the text, not necessarily correct).
        """
        labels = torch.zeros(len(tokens), dtype=int)
        labels[torch.where(tokens == self.de_id)] = self.de_label
        labels[torch.where(tokens == self.De_id)] = self.de_label
        labels[torch.where(tokens == self.dem_id)] = self.dem_label
        labels[torch.where(tokens == self.Dem_id)] = self.dem_label
        labels[torch.where(tokens == self.det_id)] = self.det_label
        labels[torch.where(tokens == self.Det_id)] = self.det_label
        labels[torch.where(tokens == self.enda_id)] = self.enda_label
        labels[torch.where(tokens == self.Enda_id)] = self.enda_label
        labels[torch.where(tokens == self.anda_id)] = self.anda_label
        labels[torch.where(tokens == self.Anda_id)] = self.anda_label

        return labels

    def ids_to_lowercase(self, tokens):
        """
        Convert cased token ids to lowercase version of id.
        """
        De_index = torch.where(tokens == self.De_id)
        Dem_index = torch.where(tokens == self.Dem_id)
        tokens[De_index] = self.de_id
        tokens[Dem_index] = self.dem_id

        Det_index = torch.where(tokens == self.Det_id)
        Enda_index = torch.where(tokens == self.Enda_id)
        Anda_index = torch.where(tokens == self.Anda_id)
        tokens[Det_index] = self.det_id
        tokens[Enda_index] = self.enda_id
        tokens[Anda_index] = self.anda_id

        return tokens

    def remove_labels_with_suffix(self, tokenized_text):
        """
        Sometimes a long word including de/dem/det/enda/ända is split into two tokens.
        If there is a suffix (i.e. the second token starts with ##), remove the label.

        E.g. "demoiselle" is split into "dem", "##ois" "##elle" ...
        """

        # Get the indices of the tokens where label != 0
        label_indices = torch.where(tokenized_text["labels"] != 0)[0]
        label_indices = label_indices.tolist()
        for label_index in label_indices:
            # Remove prediction and label if the word that immediately follows starts with ##
            # (i.e. if the word is split into two tokens)
            try:
                next_token_id = tokenized_text["input_ids"][0][label_index + 1]
                next_token = self.tokenizer.convert_ids_to_tokens(next_token_id.tolist())
                if next_token.startswith("##"):
                    tokenized_text["labels"][label_index] = 0
            except IndexError:
                continue

        return tokenized_text

    def character_span_to_token_mapping(self, tokenized_text, start_char, end_char):
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

    def __getitem__(self, idx):
        tokenized_text = self.tokenizer(self.inputs[idx], truncation=True, max_length=512, return_tensors="pt")
        tokens = tokenized_text["input_ids"][0]
        tokens = self.ids_to_lowercase(tokens)
        labels = self.get_written_word(tokens)
        # label_token_index = self.character_span_to_token_mapping(
        #     tokenized_text, self.begin_chars[idx], self.end_chars[idx]
        # )

        tokenized_text["input_ids"][0] = tokens
        tokenized_text["labels"] = labels
        # tokenized_text["label_token_index"] = label_token_index

        tokenized_text = self.remove_labels_with_suffix(tokenized_text)

        return tokenized_text


def custom_collate_fn(data):
    tokens = [sample["input_ids"][0] for sample in data]
    attention_masks = [sample["attention_mask"][0] for sample in data]
    labels = [sample["labels"] for sample in data]
    # label_token_index = [sample["label_token_index"] for sample in data]
    # sentences = [sample["sentences"] for sample in data]

    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    # labels = torch.stack(labels)  # List of B 1-length vectors to single vector of dimension B

    # label_token_index = torch.tensor(label_token_index)

    batch = {
        "input_ids": padded_tokens,
        "attention_mask": attention_masks,
        "labels": padded_labels,
        # "label_token_index": label_token_index,
        # "sentences": sentences,
    }
    return batch
