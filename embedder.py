import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def get_text_embedding(text, model, tokenizer, device="cuda:0", max_length=384, pooling="CLS"):
    """
    Generates a semantic embedding for a financial behavior description.

    Args:
        text: A string containing the user's financial behavior description.
        model: A preloaded multilingual-e5-large-instruct model.
        tokenizer: The corresponding tokenizer for the model.
        device: Device to run inference on, either 'cpu' or 'cuda'.
        max_length: Maximum number of tokens allowed in the input sequence.
        pooling: Type of pooling to use: "CLS" or "mean".
    
    Returns: 
        A NumPy array representing the semantic embedding of the input text.
    """
    model.to(device)
    model.eval()

    # Define the instruction for the instruct model
    instruction = "Represent the financial behavior of the user as a semantic embedding"
    full_input = f"Instruct: {instruction}\nQuery: {text}"

    # Tokenize the input
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    # Move tensors to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run the model in inference mode
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]

    if pooling == "CLS":
        # Use the [CLS] token (first token) representation
        embedding = hidden_states[:, 0, :]
    elif pooling == "mean":
        # Apply mean pooling over the token dimension, excluding padding
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # shape: [batch_size, seq_len, 1]
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp(min=1)  # avoid division by zero
        embedding = sum_hidden / lengths
    
    return embedding.squeeze().cpu().numpy()