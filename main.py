import torch
from transformers import AutoTokenizer
from collections import OrderedDict
from transformer import build_transformer


def text(sentence: str, model_path: str, tokenizer_src_path: str, tokenizer_tgt_path: str, seq_len: int):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizers from Hugging Face
    tokenizer_src = AutoTokenizer.from_pretrained(tokenizer_src_path)
    tokenizer_tgt = AutoTokenizer.from_pretrained(tokenizer_tgt_path)

    # Handle missing special tokens
    bos_token_id = tokenizer_src.bos_token_id or tokenizer_src.cls_token_id or tokenizer_src.pad_token_id
    eos_token_id = tokenizer_src.eos_token_id or tokenizer_src.sep_token_id or tokenizer_src.pad_token_id
    pad_token_id = tokenizer_src.pad_token_id

    if bos_token_id is None or eos_token_id is None or pad_token_id is None:
        raise ValueError("Tokenizer is missing necessary special tokens. Please set them manually.")

    # Build the transformer model (assuming build_transformer is defined)
    model = build_transformer(tokenizer_src.vocab_size, tokenizer_tgt.vocab_size, seq_len, seq_len, d_model=256, N=3, h=4)
    model.to(device)
    
    # Load the pretrained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    # Remove 'module.' prefix if it exists
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    
    # Translate the sentence
    model.eval()
    with torch.no_grad():
        # Tokenize input
        source_encoded = tokenizer_src.encode_plus(
            sentence, 
            padding="max_length", 
            max_length=seq_len - 2, 
            truncation=True, 
            return_tensors="pt"
        )

        # Manually add special tokens: [BOS] + tokens + [EOS]
        source = torch.cat([
            torch.tensor([bos_token_id], dtype=torch.int64),
            source_encoded["input_ids"].squeeze(0),
            torch.tensor([eos_token_id], dtype=torch.int64)
        ], dim=0)

        # Pad to `seq_len`
        if source.size(0) < seq_len:
            pad_length = seq_len - source.size(0)
            pad_tensor = torch.full((pad_length,), pad_token_id, dtype=torch.int64)
            source = torch.cat([source, pad_tensor])
        else:
            source = source[:seq_len]
        
        # Add batch dimension
        source = source.unsqueeze(0).to(device)
        
        # Create source mask (adjust unsqueeze if your model expects a different shape)
        source_mask = (source != pad_token_id).unsqueeze(1).int().to(device)
        
        # Compute encoder output
        encoder_output = model.encode(source, source_mask)
        
        # Initialize decoder input with [BOS] token
        decoder_input = torch.tensor([[bos_token_id]], dtype=torch.int64).to(device)
        
        # Generate tokens one by one
        while decoder_input.size(1) < seq_len:
            # Create causal mask for decoder input
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).int().to(device)
            
            # Decode
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            
            # Get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            # Append the next token
            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
            
            # Stop if [EOS] token is generated
            if next_word.item() == eos_token_id:
                break
        
    # Convert token IDs to text, skipping special tokens
    generated_text = tokenizer_tgt.decode(decoder_input.squeeze(0).tolist(), skip_special_tokens=True)
    
    # Remove "[unused0] " from the generated text
    generated_text = generated_text.replace("[unused0] ", "")
    
    return generated_text

