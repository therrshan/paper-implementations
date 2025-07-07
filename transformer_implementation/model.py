import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import TransformerEmbedding
from layers import EncoderLayer, DecoderLayer, LayerNorm


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.encoder_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len)
        self.decoder_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len)
        
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.dropout(self.encoder_embedding(src))
        enc_output = self.encoder(src_emb, src_mask)
        
        tgt_emb = self.dropout(self.decoder_embedding(tgt))
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        return output
    
    def encode(self, src, src_mask=None):
        src_emb = self.dropout(self.encoder_embedding(src))
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        tgt_emb = self.dropout(self.decoder_embedding(tgt))
        return self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)