import torch  # type: ignore
import torch.nn as nn  # type: ignore
from tqdm import trange, tqdm  # type: ignore
from torch.nn.utils.rnn import pad_sequence  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os  # type: ignore

from pig_latin_sentences import PigLatinSentences
from positional_encoding import PositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
num_tokens = 30
emb_dim = 100
batch_size = 100
lr = 1e-4
num_epochs = 200

# Character to integer mapping
alphabets = "abcdefghijklmnopqrstuvwxyz"
char_to_idx = {}
idx = 0
for char in alphabets:
    char_to_idx[char] = idx
    idx += 1
char_to_idx[' '] = idx
char_to_idx['<sos>'] = idx + 1
char_to_idx['<eos>'] = idx + 2
char_to_idx['<pad>'] = idx + 3

# Reverse mapping; integer to character
idx_to_char = {}
for char, idx in char_to_idx.items():
    idx_to_char[idx] = char

# Decoding function
@torch.no_grad()
def decode_output(output_logits, expected_words, idx_to_char):
    out_words = output_logits.argmax(2).detach().cpu().numpy()
    expected_words = expected_words.detach().cpu().numpy()
    
    out_decoded = []
    exp_decoded = []
    pad_pos = char_to_idx['<pad>']
    
    for i in range(output_logits.size(1)):  # Loop over batch_size
        out_decoded.append("".join([idx_to_char[idx] for idx in out_words[:, i] if idx != pad_pos]))
        exp_decoded.append("".join([idx_to_char[idx] for idx in expected_words[:, i] if idx != pad_pos]))

    return out_decoded, exp_decoded

# Datasets
train_dataset = PigLatinSentences("train", char_to_idx)
val_dataset = PigLatinSentences("val", char_to_idx)
test_dataset = PigLatinSentences("test", char_to_idx)

# Define embedding; padding_index does not contribute to the gradient
embedding = nn.Embedding(num_tokens, emb_dim, padding_idx=char_to_idx['<pad>'])
embedding = embedding.to(device)

# Collate function; no embedding here! just padding
def collate_fn(batch):
    eng_seqs, pig_seqs = zip(*batch)
    
    pad_idx = char_to_idx['<pad>']
    
    eng_padded = pad_sequence(eng_seqs, padding_value=pad_idx)
    pig_padded = pad_sequence(pig_seqs, padding_value=pad_idx)
    output_padded = pig_padded.clone()
    
    return eng_padded, pig_padded, output_padded  # shape: [seq_len, batch_size]

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, collate_fn=collate_fn)

# Create Transformer model
model = nn.Transformer(
    d_model=emb_dim,
    nhead=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
)
model = model.to(device)  # I/O shapes are [seq_len, batch_size, emb_dim]

# Create decoder; A linear layer mapping from embedding space to vocabulary logits
decoder = nn.Linear(emb_dim, num_tokens)
decoder = decoder.to(device)

# Positional encoder
pos_enc = PositionalEncoding(emb_dim)

# Get all parameters to optimize + Create optimizer
params = list(embedding.parameters()) + list(model.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=lr)

# Set up loss functions
mse_criterion = nn.MSELoss()  # For predicted embeddings
ce_criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])  # For predicted characters

def compare_outputs(output_text, expected_text):
    correct = 0
    for i in range(len(output_text)):
        out = output_text[i]
        exp = expected_text[i]
        exp = exp.split("<sos>")[1] # Remove <sos>
        # Remove <eos>
        if "<eos>" in out:
            out = out.split("<eos>")[0]
        exp = exp.split("<eos>")[0]

        if out == exp:
            correct += 1
    
    return correct

def train_one_epoch(epoch, num_epochs):
    avg_mse_loss = 0
    avg_ce_loss = 0
    total = 0
    correct = 0
    num_samples = 0
    
    model.train()
    for input_seq, target_seq, target_words in tqdm(train_loader, leave=False, desc=f"Train epoch {epoch+1}/{num_epochs}"):
        input_seq = input_seq.to(device)  # shape: [src_seq_len, batch_size]
        target_seq = target_seq.to(device)  # shape: [tgt_seq_len, batch_size]
        target_words = target_words.to(device)  # shape: [tgt_seq_len, batch_size]
        
        # 1. Get the input and target embeddings
        src_emb = embedding(input_seq)  # shape: [src_seq_len, batch_size, emb_dim]
        tgt_emb = embedding(target_seq)  # shape: [tgt_seq_len, batch_size, emb_dim]
        
        # 2. Pass them through the positional encodings
        src_emb = pos_enc(src_emb)
        tgt_emb = pos_enc(tgt_emb)
        
        # 3. Create the src_mask and tgt_mask
        tgt_mask = model.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)
        src_mask = None
        
        # 4. Pass the input and target embeddings through the model.
        output_emb = model(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        # output_emb: shape [tgt_seq_len, batch_size, emb_dim]
        
        # 5. Pass the output embeddings through the decoder.
        output_logits = decoder(output_emb)
        # output_logits: shape [tgt_seq_len, batch_size, num_tokens]
        
        # 6. Calculate the MSE loss between the output embeddings and the
        # target embeddings. Remember to use the target embeddings without
        # the positional encoding.
        # target_emb_raw = embedding(target_seq[1:])  # Discards the first token <sos>
        # output_emb_trimmed = output_emb[:-1]  # Discard the last predicted token
        mse_loss = mse_criterion(
            embedding(target_seq[1:]),  # Discards the first token <sos>; No pos_enc
            output_emb[:-1]  # Discard the last predicted token
        )
        
        # 7. Calculate the CE loss between the output logits and the target
        # words. Remember to reshape the output logits and target words to
        # remove the padding tokens.
        target_timmed = target_words[1:]  # Discard the first token <sos>
        logits_trimmed = output_logits[:-1]  # Discard the last predicted token
        ce_loss = ce_criterion(
            logits_trimmed.reshape(-1, num_tokens),  # Reshape to [tgt_seq_len * batch_size, num_tokens]
            target_timmed.reshape(-1)  # Reshape to [tgt_seq_len * batch_size]
        )
        
        # 8. Total loss: add MSE and CE losses, then backpropagate.
        loss = mse_loss + ce_loss
        optimizer.zero_grad()
        loss.backward()
        
        # 9. Update the parameters.
        optimizer.step()
        
        # Update losses
        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total += 1
        
        # Calculate accuracy using the decode function
        with torch.no_grad():
            output_text, expected_text = decode_output(output_logits, target_words, idx_to_char)
            correct += compare_outputs(output_text, expected_text)
            num_samples += len(output_text)
        
    # Display the decoded outputs only for the last step of each epoch
    rand_idx = [_.item() for _ in torch.randint(0, len(output_text),
                                                (min(10, len(output_text)),))]
    print(f"----{epoch+1}/{num_epochs}----")
    for i in rand_idx:
        out_ = output_text[i]
        exp_ = expected_text[i]
        print(f"Train Output:   \"{out_}\"")
        print(f"Train Expected: \"{exp_}\"")
        print("----"*40)

    return avg_mse_loss / total, avg_ce_loss / total, correct / num_samples
        
@torch.no_grad()
def validate(epoch, num_epochs):
    avg_mse_loss = 0
    avg_ce_loss = 0
    total = 0
    correct = 0
    num_samples = 0
    
    model.eval()
    for input_seq, target_seq, target_words in tqdm(val_loader, leave=False, desc=f"Val epoch {epoch+1}/{num_epochs}"):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        target_words = target_words.to(device)
        
        batch_size = input_seq.size(1)
        max_tgt_len = target_seq.size(0)
        
        src_emb = embedding(input_seq)
        src_emb = pos_enc(src_emb)
        
        # Initialize the target sequence with <sos> token
        seq_out = torch.full((1, batch_size), char_to_idx['<sos>'],
                             dtype=torch.long, device=device)
        
        # Autoregressive generation
        for _ in range(max_tgt_len - 1):  # Generate until max length reached
            tgt_emb = embedding(seq_out)
            tgt_emb = pos_enc(tgt_emb)
            
            tgt_mask = model.generate_square_subsequent_mask(seq_out.size(0)).to(device)
            
            output_emb = model(src_emb, tgt_emb, tgt_mask=tgt_mask)
            
            output_logits = decoder(output_emb)
            next_token = output_logits[-1].argmax(dim=1, keepdim=True).reshape(1, batch_size)
            
            seq_out = torch.cat([seq_out, next_token], dim=0)
        
        # Calculate the MSE and CE losses
        logits_trimmed = output_logits
        targets_trimmed = target_words[1:]
        ce_loss = ce_criterion(
            logits_trimmed.reshape(-1, num_tokens),
            targets_trimmed.reshape(-1)
        )
        
        output_emb_trimmed = output_emb
        target_emb_raw = embedding(target_seq[1:])
        mse_loss = mse_criterion(output_emb_trimmed, target_emb_raw)
        
        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total += 1
        
        # Decode and compare
        output_text, expected_text = decode_output(
            output_logits, target_words, idx_to_char
        )
        correct += compare_outputs(output_text, expected_text)
        num_samples += len(output_text)
    
    # Display a few random decoded outputs from the final batch.
    rand_idx = [i.item() for i in torch.randint(0, len(output_text), (min(10, len(output_text)),))]
    for i in rand_idx:
        print(f"Val Output:   \"{output_text[i]}\"")
        print(f"Val Expected: \"{expected_text[i]}\"")
        print("----" * 40)
    
    return avg_mse_loss / total, avg_ce_loss / total, correct / num_samples

# Training and validation loop
epoch_list = []
train_mse_loss_list = []
train_ce_loss_list = []
train_acc_list = []
val_mse_loss_list = []
val_ce_loss_list = []
val_acc_list = []

for epoch in trange(num_epochs):
    train_mse_loss, train_ce_loss, train_acc = train_one_epoch(epoch, num_epochs)
    val_mse_loss, val_ce_loss, val_acc = validate(epoch, num_epochs)
    train_mse_loss_list.append(train_mse_loss)
    train_ce_loss_list.append(train_ce_loss)
    train_acc_list.append(train_acc)
    val_mse_loss_list.append(val_mse_loss)
    val_ce_loss_list.append(val_ce_loss)
    val_acc_list.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}, train accuracy: {train_acc:.2f}, val accuracy: {val_acc:.2f}")
    print(f"Train MSE loss: {train_mse_loss:.3f}, Train CE loss: {train_ce_loss:.3f}")
    print(f"Val MSE loss: {val_mse_loss:.3f}, Val CE loss: {val_ce_loss:.3f}")
    print("----" * 40)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].plot(np.arange(len(train_ce_loss_list)), np.array(train_ce_loss_list) + np.array(train_mse_loss_list), label="Train")
    axs[0, 0].plot(np.arange(len(val_ce_loss_list)), np.array(val_ce_loss_list) + np.array(val_mse_loss_list), label="Val")
    axs[0, 0].legend()
    axs[0, 0].set_title("Total Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_yscale("log")

    axs[0, 1].plot(range(len(train_acc_list)), train_acc_list, label="Train")
    axs[0, 1].plot(range(len(val_acc_list)), val_acc_list, label="Val")
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy (%)")

    axs[1, 0].plot(range(len(train_mse_loss_list)), train_mse_loss_list, label="Train")
    axs[1, 0].plot(range(len(val_mse_loss_list)), val_mse_loss_list, label="Val")
    axs[1, 0].legend()
    axs[1, 0].set_title("MSE Loss")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].set_yscale("log")

    axs[1, 1].plot(range(len(train_ce_loss_list)), train_ce_loss_list, label="Train")
    axs[1, 1].plot(range(len(val_ce_loss_list)), val_ce_loss_list, label="Val")
    axs[1, 1].legend()
    axs[1, 1].set_title("CE Loss")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].set_yscale("log")

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/q2_results.png", dpi=300)
    plt.close()

# train_mse_loss_list = np.array(train_mse_loss_list)
# train_ce_loss_list = np.array(train_ce_loss_list)
# train_acc_list = np.array(train_acc_list)*100
# val_mse_loss_list = np.array(val_mse_loss_list)
# val_ce_loss_list = np.array(val_ce_loss_list)
# val_acc_list = np.array(val_acc_list)*100

# fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# axs[0, 0].plot(np.arange(num_epochs), train_ce_loss_list + train_mse_loss_list, label="Train")
# axs[0, 0].plot(np.arange(num_epochs), val_ce_loss_list + val_mse_loss_list, label="Val")
# axs[0, 0].legend()
# axs[0, 0].set_title("Total Loss")
# axs[0, 0].set_xlabel("Epoch")
# axs[0, 0].set_ylabel("Loss")
# axs[0, 0].set_yscale("log")

# axs[0, 1].plot(np.arange(num_epochs), train_acc_list, label="Train")
# axs[0, 1].plot(np.arange(num_epochs), val_acc_list, label="Val")
# axs[0, 1].legend()
# axs[0, 1].set_title("Accuracy")
# axs[0, 1].set_xlabel("Epoch")
# axs[0, 1].set_ylabel("Accuracy (%)")

# axs[1, 0].plot(np.arange(num_epochs), train_mse_loss_list, label="Train")
# axs[1, 0].plot(np.arange(num_epochs), val_mse_loss_list, label="Val")
# axs[1, 0].legend()
# axs[1, 0].set_title("MSE Loss")
# axs[1, 0].set_xlabel("Epoch")
# axs[1, 0].set_ylabel("Loss")
# axs[1, 0].set_yscale("log")

# axs[1, 1].plot(np.arange(num_epochs), train_ce_loss_list, label="Train")
# axs[1, 1].plot(np.arange(num_epochs), val_ce_loss_list, label="Val")
# axs[1, 1].legend()
# axs[1, 1].set_title("CE Loss")
# axs[1, 1].set_xlabel("Epoch")
# axs[1, 1].set_ylabel("Loss")
# axs[1, 1].set_yscale("log")

# fig.tight_layout()
# os.makedirs("plots", exist_ok=True)
# fig.savefig("plots/q2_results.png", dpi=300)
# plt.close()

print("Final accuracy")
print(f"Train: {train_acc_list[-1]:1.2f}")
print(f"Val: {val_acc_list[-1]:1.2f}")
print("Final losses")
print(f"Train MSE: {train_mse_loss_list[-1]:1.3f}")
print(f"Train CE: {train_ce_loss_list[-1]:1.3f}")
print(f"Val MSE: {val_mse_loss_list[-1]:1.3f}")

save_dict = {"transformer": model.state_dict(),
             "decoder": decoder.state_dict(),
             "embeddings": embedding.state_dict()}
torch.save(save_dict, "q2_model.pt")