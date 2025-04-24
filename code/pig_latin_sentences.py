import json
import torch  # type: ignore

class PigLatinSentences(torch.utils.data.Dataset):
    def __init__(self, split, char_to_idx):
        self.char_to_idx = char_to_idx
        self.english_sentences = []
        self.pig_latin_sentences = []
        self.split = split
        
        if split == 'train':
            file_path = '../starter_code/data/reviews_pig_latin_data_train.txt'
        elif split == 'val':
            file_path = '../starter_code/data/reviews_pig_latin_data_val.txt'
        elif split == 'test':
            file_path = '../starter_code/data/reviews_pig_latin_data_test.txt'
        else:
            raise ValueError(f"Unsupported split: {split}")
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.english_sentences.append(item['original'])
                if split == 'test':
                    self.pig_latin_sentences.append("")
                else:
                    self.pig_latin_sentences.append(item['pig_latin'])

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        eng = self.english_sentences[idx]
        pig = self.pig_latin_sentences[idx]
        
        eng_tokens = ["<sos>"] + list(eng) + ["<eos>"]
        eng_idx = [self.char_to_idx[char] for char in eng_tokens]
        
        if self.split == 'test':
            pig_idx = []
            return torch.tensor(eng_idx, dtype=torch.long)
        else:
            pig_tokens = ["<sos>"] + list(pig) + ["<eos>"]
            pig_idx = [self.char_to_idx[char] for char in pig_tokens]
        
        return (torch.tensor(eng_idx, dtype=torch.long),
                torch.tensor(pig_idx, dtype=torch.long))
