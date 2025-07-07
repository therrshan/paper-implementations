# Attention Is All You Need - Transformer Implementation

This is a modular implementation of the Transformer architecture from the groundbreaking paper "Attention Is All You Need" for neural machine translation using PyTorch.

## 📋 Executive Summary

This project implements the Transformer model that revolutionized NLP by replacing recurrent layers entirely with self-attention mechanisms. The implementation demonstrates German-to-English translation on the Multi30k dataset, showcasing the key architectural innovations that became the foundation for models like BERT and GPT.

**Key Metrics:**
- **Model Size**: ~14M parameters (~56MB)
- **Dataset**: Multi30k (30,000 German-English translation pairs)
- **Classes**: ~8K German tokens, ~6K English tokens
- **Expected Performance**: ~25-35 BLEU score

## 📊 Training Results

After training, you can monitor detailed training progress with loss and perplexity metrics displayed during training.

**Performance Metrics:**
- **Total Epochs**: 20
- **Best Validation Loss**: ~2.5
- **Best Perplexity**: ~12-15
- **Training Time**: ~45-60 minutes (GPU) / ~3-4 hours (CPU)

## 🚀 Quick Start

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install torch numpy spacy
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### Training

To train the Transformer model:
```bash
python main.py
```

This will:
- Load Multi30k dataset from the data folder
- Train Transformer for 20 epochs
- Save the model as `transformer_multi30k.pt`
- Save vocabularies as `multi30k_vocabs.pkl`
- Display training progress with batch-level updates
- Show German-to-English translation examples

### Inference

To test the trained model:
```bash
python inference.py
```

## 🔗 References

1. Vaswani, A., et al. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.
2. Multi30k Dataset: Elliott, D., et al. (2016). Multi30k: Multilingual English-German image descriptions. *ACL Workshop on Vision and Language*.
3. The Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html