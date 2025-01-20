import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, output_dim):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, d_model)  # Embedding layer
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, output_dim)  # Final output layer
    
    def forward(self, src, tgt):
        # src: source sequence, tgt: target sequence
        # src = self.embedding(src) * math.sqrt(src.size(2))  # Scale by sqrt(d_model)
        # tgt = self.embedding(tgt) * math.sqrt(tgt.size(2))  # Same for target

        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)

        
        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        
        # Pass through transformer encoder and decoder
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)

        
        
        # Final output layer
        output = self.fc_out(output)
        return output


# Hyperparameters
input_dim = 10000  # Vocabulary size
output_dim = 10000  # Vocabulary size (for language generation tasks)
d_model = 512  # Embedding dimension
nhead = 8  # Number of attention heads
num_encoder_layers = 6
num_decoder_layers = 6
batch_size = 32
seq_length = 30

# Model initialization
model = Transformer(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, output_dim)

# Example random input (batch_size x seq_length)
src = torch.randint(0, input_dim, (seq_length, batch_size))
tgt = torch.randint(0, input_dim, (seq_length, batch_size))



# Forward pass
output = model(src, tgt)

# Print the output shape (should be [seq_length, batch_size, output_dim])
print(src.shape)
print(output.shape)
