import re
import os
import torch
import torch.nn as nn
import streamlit as st
from tokenizers import Tokenizer
import math

# Setting page configuration
st.set_page_config(
    page_title="English to Arabic Translator",
    page_icon="",
    layout="centered"
)

# 1. Text Preprocessing

def clean_english_input(text):
    """Applies the exact same cleaning rules used during training."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '<URL>', text)
    text = re.sub(r'-{2,}', '-', text)   # Standardize hyphens
    text = re.sub(r'\.{2,}', '...', text) # Standardize ellipses
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# 2. Model Architecture Skeleton

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape [max_len, d_model] to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create a column vector of positions [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the denominator for the sine/cosine arguments
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as a buffer so it saves with the model state but isn't updated by the optimizer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, d_model]
        """
        # Add the positional encoding up to the current sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, 
                 nhead, src_vocab_size, tgt_vocab_size, dim_feedforward, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        
        # Native PyTorch Transformer
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Crucial for our DataLoader configuration
        )
        
        # Linear layer to map the decoder output back to the target vocabulary size
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        
        # Distinct embedding layers for source (English) and target (Arabic)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        # Apply embeddings and positional encoding, scaling by sqrt(d_model) as per the paper
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.transformer.d_model))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg) * math.sqrt(self.transformer.d_model))
        
        # Pass everything through the Transformer
        outs = self.transformer(
            src_emb, tgt_emb, 
            src_mask=src_mask, tgt_mask=tgt_mask, 
            memory_mask=None, # Usually not needed for standard seq2seq
            src_key_padding_mask=src_padding_mask, 
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Project the output to the target vocabulary
        return self.generator(outs)

# Masking Logic

def generate_square_subsequent_mask(sz):
    """
    Creates a look-ahead mask for the decoder to prevent it from "seeing" future tokens.
    Returns a matrix filled with 0s and -inf.
    """
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    """
    Generates both the padding masks and the look-ahead mask.
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # Look-ahead mask for the target sequence
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    
    # The source sequence doesn't need a look-ahead mask, so it's all zeros
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    # Padding masks to ignore [PAD] tokens
    src_padding_mask = (src == src_pad_idx)
    tgt_padding_mask = (tgt == tgt_pad_idx)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# 3. Cached Resource Loading (Optimizes Performance)

@st.cache_resource
def load_resources():
    """Loads tokenizers and model weights into memory exactly once."""
    # Load Tokenizers
    if not os.path.exists(r"model_artifacts\en_tokenizer.json") or not os.path.exists(r"model_artifacts\ar_tokenizer.json"):
        st.error("Tokenizer files ('en_tokenizer.json', 'ar_tokenizer.json') not found in the current directory.")
        return None, None, None

    en_tokenizer = Tokenizer.from_file(r"model_artifacts\en_tokenizer.json")
    ar_tokenizer = Tokenizer.from_file(r"model_artifacts\ar_tokenizer.json")
    
    # Initialize Model Hyperparameters (Must match Part 3 and Part 10)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hparams
    VOCAB_SIZE = 10000
    EMB_SIZE = 256
    NHEAD = 4
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    # Reconstruct Model
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE, 
        NHEAD,
        VOCAB_SIZE,
        VOCAB_SIZE,
        FFN_HID_DIM
    ).to(DEVICE)
    
    # Load Weights
    if os.path.exists(r"model_artifacts\best_transformer_arabic.pth"):
        # Map storage to CPU if GPU isn't available to prevent runtime crashes
        state_dict = torch.load(r"model_artifacts\best_transformer_arabic.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        st.error("'best_transformer_arabic.pth' weights file not found.")
        return None, None, None, None
        
    return model, en_tokenizer, ar_tokenizer, DEVICE

# Load resources smoothly
model, en_tokenizer, ar_tokenizer, DEVICE = load_resources()


# 4. Greedy Decoding Inference Loop

def generate_translation(model, src_sentence, en_tokenizer, ar_tokenizer, device, max_len=30):
    """Performs token-by-token generation using the trained Transformer."""
    # Clean and encode source text
    cleaned_src = clean_english_input(src_sentence)
    encoded_src_ids = en_tokenizer.encode(cleaned_src).ids
    
    # Get Special Token IDs
    EN_SOS_IDX = en_tokenizer.token_to_id("[SOS]")
    EN_EOS_IDX = en_tokenizer.token_to_id("[EOS]")
    AR_SOS_IDX = ar_tokenizer.token_to_id("[SOS]")
    AR_EOS_IDX = ar_tokenizer.token_to_id("[EOS]")
    
    # Wrap the English IDs with [SOS] and [EOS] and cast to a 2D tensor (Batch size 1)
    src_tensor = torch.tensor([[EN_SOS_IDX] + encoded_src_ids + [EN_EOS_IDX]], dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        # Step A: Pre-calculate Encoder Memory
        # Create a dummy mask of zeros for the source sequence
        src_mask = torch.zeros((src_tensor.shape[1], src_tensor.shape[1]), device=device).type(torch.bool)
        # We don't need a padding mask for a single sentence without padding, but we pass an all-False tensor to be safe
        src_padding_mask = torch.zeros((1, src_tensor.shape[1]), device=device).type(torch.bool)
        
        # 1. Manually embed and encode the source sequence
        src_emb = model.positional_encoding(model.src_tok_emb(src_tensor) * (model.transformer.d_model ** 0.5))
        memory = model.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        # Step B: Loop to generate tokens sequentially
        ys = torch.tensor([[AR_SOS_IDX]], dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # 2. Create autoregressive causal mask for decoder
            tgt_mask = (torch.triu(torch.ones((ys.shape[1], ys.shape[1]), device=device)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
            
            # 3. Manually embed the current target sequence and decode
            tgt_emb = model.positional_encoding(model.tgt_tok_emb(ys) * (model.transformer.d_model ** 0.5))
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
            
            # 4. Project the last generated token to the vocabulary space
            logits = model.generator(out[:, -1])
            
            # Select token with highest probability
            _, next_word = torch.max(logits, dim=1)
            next_word_id = next_word.item()
            
            # Append token to sequence
            ys = torch.cat([ys, torch.tensor([[next_word_id]], device=device)], dim=1)
            
            # Break if End-of-Sentence token is reached
            if next_word_id == AR_EOS_IDX:
                break
                
    # Decode token IDs back to a human-readable Arabic string
    generated_ids = ys.squeeze().tolist()
    
    # Handle edge case if it only generates a single token
    if isinstance(generated_ids, int):
        generated_ids = [generated_ids]
        
    # skip_special_tokens=True automatically strips [SOS] and [EOS] from the final text
    translation = ar_tokenizer.decode(generated_ids, skip_special_tokens=True)
    return translation


# 5. Streamlit UI Elements

st.title("Conversational NMT Dashboard")
st.subheader("English to Modern Standard Arabic (MSA) Translation")
st.write("This lightweight model is optimized for short conversational phrases and dialogue configurations under 30 words.")

# Layout Configuration
st.sidebar.header("Model Specifications")
st.sidebar.markdown("""
- **Architecture:** 3-Layer Transformer
- **Embedding Dim ($d_{model}$):** 256
- **Attention Heads:** 4
- **Vocabulary Size:** 10,000 (BPE)
- **Target Constraint:** Max 30 words
""")

# User input text area
user_input = st.text_area(
    "Enter English Text:", 
    placeholder="Type something conversational here (e.g., 'Where are you going tomorrow?')...",
    max_chars=300
)

# Trigger inference when button is pressed
if st.button("Translate", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to translate.")
    elif model is None:
        st.error("System error: Resources failed to initialize. Check the warnings/errors above.")
    else:
        # Check constraints gracefully
        word_count = len(user_input.split())
        if word_count > 30:
            st.warning(f"Your input has {word_count} words. The model is strictly optimized for sentences ≤ 30 words. Results may degrade.")
            
        with st.spinner("Decoding tokens..."):
            try:
                # Run the translation loop
                output_translation = generate_translation(model, user_input, en_tokenizer, ar_tokenizer, DEVICE)
                
                # Display Output Container
                st.markdown("### Translation Output:")
                st.success(output_translation)
            except Exception as e:
                st.error(f"Inference Runtime Error: {e}")