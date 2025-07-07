import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import os
import spacy


class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_count = Counter()
        self.n_words = 4
        
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.word_count[word] += 1
            
    def build_vocab(self, min_freq=2):
        for word, count in self.word_count.items():
            if count >= min_freq:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
                
    def encode(self, sentence):
        tokens = ['<sos>']
        for word in sentence.split():
            tokens.append(self.word2idx.get(word, self.word2idx['<unk>']))
        tokens.append('<eos>')
        return [self.word2idx[tokens[0]]] + tokens[1:-1] + [self.word2idx[tokens[-1]]]
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx['<eos>']:
                break
            if idx not in [self.word2idx['<pad>'], self.word2idx['<sos>']]:
                words.append(self.idx2word.get(idx, '<unk>'))
        return ' '.join(words)


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_encoded = self.src_vocab.encode(self.src_sentences[idx])
        tgt_encoded = self.tgt_vocab.encode(self.tgt_sentences[idx])
        
        src_padded = self.pad_sequence(src_encoded)
        tgt_padded = self.pad_sequence(tgt_encoded)
        
        return {
            'src': torch.LongTensor(src_padded),
            'tgt': torch.LongTensor(tgt_padded)
        }
    
    def pad_sequence(self, seq):
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [self.src_vocab.word2idx['<pad>']] * (self.max_len - len(seq))
        return seq


def load_data(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip().lower() for line in f.readlines()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = [line.strip().lower() for line in f.readlines()]
    
    return src_sentences, tgt_sentences


def tokenize_data(sentences, spacy_model):
    tokenized = []
    for sent in sentences:
        tokens = [tok.text for tok in spacy_model.tokenizer(sent)]
        tokenized.append(' '.join(tokens))
    return tokenized


def create_vocabs(src_sentences, tgt_sentences, min_freq=2):
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    for sentence in src_sentences:
        src_vocab.add_sentence(sentence)
    
    for sentence in tgt_sentences:
        tgt_vocab.add_sentence(sentence)
    
    src_vocab.build_vocab(min_freq)
    tgt_vocab.build_vocab(min_freq)
    
    return src_vocab, tgt_vocab


def main():
    # Try GPU first, fallback to CPU if error
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            # Test GPU availability
            torch.cuda.empty_cache()
            test_tensor = torch.zeros(1).to(device)
            del test_tensor
    except RuntimeError as e:
        print(f"GPU error: {e}")
        print("Falling back to CPU...")
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    if not os.path.exists('data/train.de'):
        print("ERROR: Multi30k dataset not found!")
        print("Please download the dataset and place files in the 'data' folder:")
        print("  - train.de, train.en")
        print("  - val.de, val.en")
        print("  - test.de, test.en")
        return
    
    print("\nLoading spaCy models for tokenization...")
    try:
        spacy_de = spacy.load('de_core_news_sm')
    except:
        print("Installing German spaCy model...")
        os.system('python -m spacy download de_core_news_sm')
        spacy_de = spacy.load('de_core_news_sm')
    
    try:
        spacy_en = spacy.load('en_core_web_sm')
    except:
        print("Installing English spaCy model...")
        os.system('python -m spacy download en_core_web_sm')
        spacy_en = spacy.load('en_core_web_sm')
    
    print("\nLoading Multi30k dataset...")
    src_train, tgt_train = load_data('data/train.de', 'data/train.en')
    src_val, tgt_val = load_data('data/val.de', 'data/val.en')
    
    print("Tokenizing data...")
    src_train = tokenize_data(src_train, spacy_de)
    tgt_train = tokenize_data(tgt_train, spacy_en)
    src_val = tokenize_data(src_val, spacy_de)
    tgt_val = tokenize_data(tgt_val, spacy_en)
    
    print(f"Train samples: {len(src_train)}")
    print(f"Validation samples: {len(src_val)}")
    print(f"Example pair:")
    print(f"  German: {src_train[0]}")
    print(f"  English: {tgt_train[0]}")
    
    print("\nBuilding vocabularies...")
    src_vocab, tgt_vocab = create_vocabs(src_train + src_val, tgt_train + tgt_val, min_freq=2)
    print(f"Source vocab size: {src_vocab.n_words}")
    print(f"Target vocab size: {tgt_vocab.n_words}")
    
    train_dataset = TranslationDataset(src_train, tgt_train, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(src_val, tgt_val, src_vocab, tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced from 32
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)     # Reduced from 32
    
    model = Transformer(
        src_vocab_size=src_vocab.n_words,
        tgt_vocab_size=tgt_vocab.n_words,
        d_model=256,  # Reduced for faster training
        n_heads=8,
        n_layers=3,   # Reduced for faster training
        d_ff=1024,    # Reduced for faster training
        max_len=100,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining model...")
    train_transformer(model, train_loader, val_loader, n_epochs=20, device=device, 
                     checkpoint_path='transformer_multi30k.pt')
    
    print("\nTesting translation...")
    model.load_state_dict(torch.load('transformer_multi30k.pt'))
    
    test_sentences = [
        "zwei hunde spielen im schnee .",
        "ein mann fährt ein blaues fahrrad auf einer straße .",
        "eine frau mit einer großen handtasche geht an einem tor vorbei .",
        "kinder spielen auf einem spielplatz ."
    ]
    
    model.eval()
    for sentence in test_sentences:
        tokens = [tok.text for tok in spacy_de.tokenizer(sentence)]
        tokenized = ' '.join(tokens)
        translation = translate_sentence(model, tokenized, src_vocab, tgt_vocab, device)
        print(f"German: {sentence}")
        print(f"English: {translation}")
        print()
    
    with open('multi30k_vocabs.pkl', 'wb') as f:
        pickle.dump({'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, f)
    
    print("Training complete! Model and vocabularies saved.")


if __name__ == "__main__":
    from attention import MultiHeadAttention
    from layers import EncoderLayer, DecoderLayer, LayerNorm, PositionwiseFeedForward
    from embeddings import TransformerEmbedding, PositionalEncoding
    from model import Transformer, Encoder, Decoder
    from utils import create_masks, create_padding_mask, create_look_ahead_mask, get_std_opt
    from train import train_transformer, train_epoch, evaluate
    from inference import translate_sentence, greedy_decode, beam_search_decode
    
    main()