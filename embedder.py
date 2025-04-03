import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def get_text_embedding(text, model, tokenizer, device="cuda:0", max_length=384):
    """
    Generates a semantic embedding for a financial behavior description using the multilingual-e5-large-instruct model.

    Args:
        text: A string containing the user's financial behavior description.
        model: A preloaded multilingual-e5-large-instruct model.
        tokenizer: The corresponding tokenizer for the model.
        device: Device to run inference on, either 'cpu' or 'cuda'.
        max_length: Maximum number of tokens allowed in the input sequence.
    
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
    
    # Extract the embedding from the CLS token
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding