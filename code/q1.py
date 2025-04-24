import torch  # type:ignore
import torch.nn as nn  # type:ignore
import torch.optim as optim  # type:ignore
from tqdm import tqdm, trange  # type:ignore
from yelp_dataset import YelpDataset  # type:ignore
import torch.nn.utils.rnn as rnn_utils  # type:ignore
import seaborn as sns  # type:ignore
import matplotlib.pyplot as plt  # type:ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
emb_dim = 50
batch_size = 64
rnn_dropout = 0.25
num_rnn_layers = 2
lr = 5e-4
num_epochs = 10

# Load the datasets
train_dataset = YelpDataset("train")
val_dataset = YelpDataset("val")
test_dataset = YelpDataset("test")

# Load the modified GloVe embeddings into nn.Embedding instance
glove_data = torch.load("../starter_code/glove/modified_glove_50d.pt",
                        weights_only=True)
words_list = list(glove_data.keys())
embeddings_list = [glove_data[word] for word in words_list]  # Shape of each element [1, emb_dim]
emb_init_tensor = torch.cat(embeddings_list, dim=0)        # Shape: [vocab_size, emb_dim]
embeddings = nn.Embedding.from_pretrained(emb_init_tensor, freeze=False)
embeddings = embeddings.to(device)

# Define the collate function
def collate_fn(batch):
    if isinstance(batch[0], tuple):  # Train or validation set
        sequences, stars = zip(*batch)
    else:  # Test set
        sequences, stars = batch, None
    
    # Convert sequences to embeddings
    embedded_sequences = [embeddings(seq.to(device)) for seq in sequences]
    lengths = [s.size(0) for s in embedded_sequences]
    
    # Pad the sequences
    padded_sequences = torch.nn.utils.rnn.pad_sequence(embedded_sequences, batch_first=True)
    
    # Pack the padded sequences
    packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(padded_sequences, lengths,
                                                                batch_first=True, enforce_sorted=False)
    
    if stars is not None:
        stars = torch.tensor(stars, dtype=torch.long).to(device)
        return packed_sequences, stars
    else:
        return packed_sequences

# Instantiate DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, collate_fn=collate_fn)

# Create the RNN model
model = nn.RNN(
    input_size=emb_dim,
    hidden_size=emb_dim,
    num_layers=num_rnn_layers,
    dropout=rnn_dropout,
    batch_first=True
)
model = model.to(device)

# Create the linear classifier
classifier = nn.Linear(emb_dim, 5)
classifier = classifier.to(device)

# Get all parameters and create an optimizer to update them
params = list(embeddings.parameters()) + list(model.parameters()) + list(classifier.parameters())
optimizer = optim.Adam(params, lr=lr)

# Create the loss function
criterion = nn.CrossEntropyLoss()

# Initialize train/val loss/acc lists
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

def train_one_epoch():
    avg_loss = 0
    num_steps = 0
    correct = 0
    total_samples = 0
    
    model.train()
    for review, stars in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Pass the packed sequences through the model
        packed_output, hidden = model(review)
        
        # Unpack the output; padded_output is of shape (batch_size, max_seq_len, emb_dim)
        padded_output, lengths = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the last output for each sequence
        batch_size = padded_output.size(0)
        lengths_tensor = torch.as_tensor(lengths, device=padded_output.device) - 1  # adjust for 0-indexing
        batch_indices = torch.arange(batch_size, device=padded_output.device)
        last_outputs = padded_output[batch_indices, lengths_tensor, :]  # Shape: [batch_size, hidden_dim]
        
        # Pass the last outputs through the classifier
        logits = classifier(last_outputs)
        
        # Compute the loss
        loss = criterion(logits, stars - 1)
        
        # Backpropagate the loss and update model parameters
        loss.backward()
        optimizer.step()
        
        # Update the running statistics.
        with torch.no_grad():
            avg_loss += loss.item()
            total_samples += stars.size(0)
            preds = torch.argmax(logits, dim=1) + 1
            correct += (preds == stars).sum().item()
            num_steps += 1
    
    avg_loss /= num_steps
    accuracy = 100 * correct / total_samples

    return avg_loss, accuracy

@torch.no_grad()
def validate():
    avg_loss = 0
    num_steps = 0
    correct = 0
    total_samples = 0
    model.eval()
    confusion_matrix = torch.zeros(5, 5)

    for review, stars in tqdm(val_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"):
        packed_output, hidden = model(review)
        
        padded_output, lengths = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        
        batch_size = padded_output.size(0)
        lengths_tensor = torch.as_tensor(lengths, device=padded_output.device) - 1
        batch_indices = torch.arange(batch_size, device=padded_output.device)
        last_outputs = padded_output[batch_indices, lengths_tensor, :]

        logits = classifier(last_outputs)
        
        loss = criterion(logits, stars - 1)

        avg_loss += loss.item()
        total_samples += stars.size(0)
        preds = torch.argmax(logits, dim=1) + 1
        correct += (preds == stars).sum().item()
        
        for i in range(stars.size(0)):
            confusion_matrix[stars[i]-1, preds[i]-1] += 1
        
        num_steps += 1

    avg_loss /= num_steps
    accuracy = 100 * correct / total_samples

    return avg_loss, accuracy, confusion_matrix

pbar = trange(num_epochs)
for epoch in pbar:
    train_loss, train_accuracy = train_one_epoch()
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)
    val_loss, val_accuracy, confusion_matrix = validate()
    val_loss_list.append(val_loss)
    val_acc_list.append(val_accuracy)

    pbar.set_postfix({"Train Loss": f"{train_loss:1.3f}", "Train Accuracy": f"{train_accuracy:1.2f}",
                      "Val Loss": f"{val_loss:1.3f}", "Val Accuracy": f"{val_accuracy:1.2f}"})

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(train_loss_list, label="Train")
    axs[0].plot(val_loss_list, label="Val")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(train_acc_list, label="Train")
    axs[1].plot(val_acc_list, label="Val")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(f"plots/q1_plot_lr{lr}_bs{batch_size}_do{rnn_dropout}.png", dpi=300, bbox_inches="tight")
    plt.close()

    sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"plots/q1_confusion_matrix_lr{lr}_bs{batch_size}_do{rnn_dropout}.png", dpi=300, bbox_inches="tight")
    plt.close()

torch.save(model.state_dict(), "q1_model.pt")
torch.save(classifier.state_dict(), "q1_classifier.pt")
torch.save(embeddings.state_dict(), "q1_embedding.pt")