import json
import torch  # type:ignore
from torch.utils.data import Dataset  # type: ignore

class YelpDataset(Dataset):
    def __init__(self, split):
        # train, val, or test
        self.split = split
        
        # Load the modified glove embeddings (Dict: word -> tensor[1, 50])
        self.glove_data = torch.load("../starter_code/glove/modified_glove_50d.pt",
                                     weights_only=True)
        
        # Extract words and embeddings from the glove dictionary
        words, embs = [], []
        for word, emb in self.glove_data.items():
            words.append(word)
            embs.append(emb)
        
        # Create a dictionary mapping from word to index
        self.word_indices = {word: idx for idx, word in enumerate(words, start=0)}
        
        # Load the Yelp dataset
        with open(f"../starter_code/data/yelp_dataset_{split}.json", "r") as f:
            data = [json.loads(line) for line in f]
        
        self.reviews = [item["review"] for item in data]
        self.stars = [item["stars"] for item in data] if split != "test" else None

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # Split the review into words
        words = self.reviews[idx].split()
        
        # Convert words to indices
        emb = torch.tensor([self.word_indices[w] for w in words], dtype=torch.long)
        
        if self.split == "test":
            return emb
        else:
            return emb, self.stars[idx]
