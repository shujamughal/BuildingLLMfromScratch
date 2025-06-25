import re
class SimpleTokenizer:
    def __init__(self, vocab):
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, text):
        # Step 1: Split text into tokens using regex
        tokens = re.split(r"(\s+|,|\.|:|;|\?|\!|\"|--|\(|\))", text)
        tokens = [t for t in tokens if t.strip()]

        # Step 2: Convert tokens to IDs, use <unk> for unknowns
        return [self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in tokens]

    def decode(self, ids):
        # Step 3: Convert IDs back to tokens
        tokens = [self.id_to_token[i] for i in ids]

        # Step 4: Merge tokens back into readable string
        return " ".join(tokens).replace(" .", ".").replace(" ,", ",")
