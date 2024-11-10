import os
import nltk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot  # For model visualization

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Read the captions file
        with open(captions_file, 'r') as f:
            self.captions = f.readlines()

        # Build a list of (image, caption) tuples
        self.imgs = []
        for line in self.captions:
            tokens = line.strip().split('\t')
            img_id_full = tokens[0].split('#')[0]
            img_id = os.path.basename(img_id_full)  # Extract filename
            caption = tokens[1]
            self.imgs.append((img_id, caption))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_id, caption = self.imgs[index]
        img_path = os.path.join(self.root_dir, 'Images', img_id)
        # Debugging line to check image paths
        # print(f"Loading image from: {img_path}")
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, caption

transform = transforms.Compose([
    transforms.Resize(256),  
    transforms.RandomCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize((0.485, 0.456, 0.406),  
                         (0.229, 0.224, 0.225))
])

dataset = Flickr8kDataset(
    root_dir='c:/Users/Xolan/Downloads/Flickr8k_Dataset/',
    captions_file='c:/Users/Xolan/Downloads/Flickr8k_Dataset/Flickr8k.token.txt',
    transform=transform
)

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4  # Starting index for new words

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] +=1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx +=1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

captions = [caption for _, caption in dataset.imgs]
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(captions)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False  # Freeze the ResNet model

        modules = list(resnet.children())[:-2]  # Remove last pooling and FC layers
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(embed_size)
        self.relu = nn.ReLU()
        self.embed_size = embed_size

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, H, W)
        features = self.adaptive_pool(features)  # (batch_size, 2048, 14, 14)
        features = self.conv(features)  # (batch_size, embed_size, 14, 14)
        features = self.bn(features)
        features = self.relu(features)
        features = features.view(features.size(0), -1, self.embed_size)  # (batch_size, num_pixels, embed_size)
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Linear layer to transform encoder output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Linear layer to transform decoder's hidden state
        self.full_att = nn.Linear(attention_dim, 1)  # Linear layer to calculate attention weights
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, decoder_dim, vocab_size, attention_dim, encoder_dim=None, dropout=0.5):
        super(DecoderRNN, self).__init__()
        if encoder_dim is None:
            encoder_dim = embed_size  # Default to embed_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # Attention Network

        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)  # LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # Initialize hidden state
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # Initialize cell state
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # Create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # Final output layer

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]

        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <EOS> position, since we've finished generating as soon as we generate <EOS>
        decode_lengths = [length.item() - 1 for length in caption_lengths]

        # Create tensors to hold word predictions and alphas
        max_dec_len = max(decode_lengths)
        predictions = torch.zeros(batch_size, max_dec_len, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_dec_len, num_pixels).to(device)

        # At each time-step, decode by attention-weighing the encoder's output based on the decoder's previous hidden state
        for t in range(max_dec_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            current_embeddings = embeddings[:batch_size_t, t, :]
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([current_embeddings, attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )  # LSTMCell

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions, decode_lengths, alphas, sort_ind

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, decoder_dim, vocab_size, attention_dim, encoder_dim, dropout=0.5):
        super(EncoderDecoder, self).__init__()
        if encoder_dim is None:
            encoder_dim = embed_size  # Default to embed_size
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, decoder_dim, vocab_size, attention_dim, encoder_dim, dropout)

    def forward(self, images, captions, caption_lengths):
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions, caption_lengths)
        return outputs

def collate_fn(data):
    data.sort(key=lambda x: len(x[1].split()), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    # Adjust caption lengths to include <SOS> and <EOS>
    caption_lengths = torch.tensor([len(caption.split()) + 2 for caption in captions]).unsqueeze(1)
    targets = [torch.tensor([vocab.stoi["<SOS>"]] + vocab.numericalize(caption) + [vocab.stoi["<EOS>"]]) for caption in captions]
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=vocab.stoi["<PAD>"])

    return images, targets, caption_lengths

def generate_caption(model, image, vocab, max_length=20):
    model.eval()
    result_caption = []
    attention_plot = []

    with torch.no_grad():
        # Get image features from encoder
        encoder_out = model.encoder(image.unsqueeze(0).to(device))
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Initialize hidden state and cell state
        h, c = model.decoder.init_hidden_state(encoder_out)

        # Start with SOS token
        input_word = torch.tensor([vocab.stoi["<SOS>"]]).to(device)

        for _ in range(max_length):
            # Get word embeddings
            embeddings = model.decoder.embedding(input_word)  # Shape: (1, embed_size)

            # Get attention weighted encoding
            attention_weighted_encoding, alpha = model.decoder.attention(encoder_out, h)
            gate = model.decoder.sigmoid(model.decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding  # Shape: (1, encoder_dim)

            # Concatenate embeddings and attention weighted encoding
            decoder_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)  # Shape: (1, embed_size + encoder_dim)

            # LSTM step
            h, c = model.decoder.decode_step(decoder_input, (h, c))

            # Generate output
            output = model.decoder.fc(h)
            predicted = output.argmax(1)

            # Save results
            result_caption.append(predicted.item())
            attention_plot.append(alpha.cpu().numpy())

            # Break if EOS token is predicted
            if predicted.item() == vocab.stoi["<EOS>"]:
                break

            # Update input word for next iteration
            input_word = predicted

    # Convert indices to words
    caption = [vocab.itos[idx] for idx in result_caption]
    return caption, attention_plot

def visualize_attention(image_path, caption, attention_plot):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    fig = plt.figure(figsize=(15, 15))
    len_cap = len(caption)

    for idx in range(len_cap):
        temp_att = attention_plot[idx].reshape(14, 14)  # Adjusted to 14x14
        temp_att = temp_att / temp_att.max()

        ax = fig.add_subplot(np.ceil(len_cap/5.), 5, idx+1)
        ax.set_title(caption[idx])
        img = ax.imshow(image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([256, 256], Image.LANCZOS)
    if transform is not None:
        image = transform(image).to(device)
    return image

if __name__ == '__main__':
    nltk.download('punkt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_size = 256
    attention_dim = 256
    decoder_dim = 512
    vocab_size = len(vocab)
    learning_rate = 1e-4
    num_epochs = 20
    save_every = 5  # Save the model every 5 epochs

    model = EncoderDecoder(embed_size, decoder_dim, vocab_size, attention_dim, encoder_dim=embed_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model.train()
    total_loss_list = []
    plt.ion()  # Turn on interactive mode for non-blocking plots

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, captions, caption_lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)
            # print(f"Caption lengths: {caption_lengths}")

            optimizer.zero_grad()

            outputs, targets, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)

            # Since we sorted captions, targets are already sorted
            targets = targets[:, 1:]  # Remove <SOS>

            # Pack the sequences
            outputs_packed = pack_padded_sequence(outputs, decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(outputs_packed, targets_packed)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}')

        avg_epoch_loss = epoch_loss / len(data_loader)
        total_loss_list.append(avg_epoch_loss)

        # Plot training loss
        plt.figure(1)
        plt.clf()
        plt.plot(total_loss_list, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.pause(0.001)  # Pause to update the plot

        # Save the model at specified intervals
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
            print(f'Model saved at epoch {epoch+1}')

    plt.ioff()  # Turn off interactive mode
    plt.show()

    # Visualize the model architecture using torchviz
    sample_image, sample_caption = next(iter(data_loader))
    sample_image = sample_image[0].unsqueeze(0).to(device)
    sample_caption = sample_caption[0].unsqueeze(0).to(device)
    sample_caption_length = caption_lengths[0].unsqueeze(0).to(device)

    # Forward pass to get the computation graph
    outputs = model(sample_image, sample_caption, sample_caption_length)
    make_dot(outputs[0], params=dict(list(model.named_parameters()))).render("model_architecture", format="png")
    print('Model architecture saved as model_architecture.png')

    # Replace with the path to your test image
    image_path = 'C:/Users/Xolan/Downloads/Flickr8k_Dataset/Images/990890291_afc72be141.jpg'
    image = load_image(image_path, transform)

    # Generate caption
    caption, attention_plot = generate_caption(model, image, vocab)

    # Remove <EOS> token and visualize
    caption = caption[:-1]  # Remove <EOS>
    visualize_attention(image_path, caption, attention_plot)
