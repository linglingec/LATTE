from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import vllm
import openai
import asyncio
import httpx
import time
import subprocess
from pathlib import Path
import os
import csv
from tqdm import tqdm

from preprocessing import preprocess_transactions, generate_prompts

def query_llm(prompt, processor, model):
    """
    Sends the prompt to the LLM and retrieves a generated response.

    Args:
        prompt (str): The prompt describing a user's transaction behavior.

    Returns:
        str: Generated financial behavior summary from the model.
    """

    # Prepare the input message in the required format
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    # Generate the response
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, 
                        return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=True)
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

# Main function: processes dataset and generates user descriptions
def generate_client_descriptions(df, user_col, time_col, amount_col, start_from_id=None, category_col=None, type_col=None, dataset_name="gender", output_path="client_descriptions.csv"):
    """
    Processes transactional data, generates prompts, and retrieves LLM-generated descriptions.

    Args:
        df (pd.DataFrame): The raw transaction dataset.
        user_col (str): The column name for user identification.
        time_col (str): The column name for timestamps.
        amount_col (str): The column name for transaction amounts.
        category_col (str, optional): The column name for transaction categories.
        type_col (str, optional): The column name for transaction types.
        dataset_name (str): The dataset name to adjust preprocessing logic.
        output_path (str): Path to the output CSV file.
    """
    processed_df = preprocess_transactions(df, user_col, time_col, amount_col, category_col, type_col, dataset_name)
    if start_from_id is not None:
        processed_df = processed_df[processed_df[user_col] > start_from_id]
    prompts = generate_prompts(processed_df, user_col, dataset_name)

    descriptions = []

    # Initialize the Gemma 3 model pipeline
    model_id = MODEL_NAME
    hf_token = HF_TOKEN

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", token=hf_token).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    # Write header to the output CSV if it doesn't exist
    if not os.path.isfile(output_path):
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "description"])
            writer.writeheader()

    for user_id, prompt in tqdm(prompts, total=len(prompts), desc="Generating LLM responses", ncols=80):
        llm_response = query_llm(prompt, processor, model)
        descriptions.append({"user_id": user_id, "description": llm_response})

        with open(output_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "description"])
            writer.writerow({"id": user_id, "description": llm_response})

# Launch vLLM Server
HF_TOKEN = "YOUR_TOKEN"
MODEL_NAME = "google/gemma-3-27b-it"

def launch_vllm_server(model_path="google/gemma-3-27b-it", port=8000, max_model_len=1500):
    """
    Starts a vLLM server via subprocess using OpenAI-compatible API.

    Args:
        model_path (str): Model path or HuggingFace model ID.
        port (int): Port to expose the API.
        max_model_len (int): Max token limit for context + generation.
    """
    env = os.environ.copy()
    env["HUGGINGFACE_TOKEN"] = HF_TOKEN  # <-- Insert your HF token

    log_file = open("vllm_server.log", "w")

    command = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--tokenizer", MODEL_NAME,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.9"
    ]

    print(f" Launching vLLM server on port {port}...")
    subprocess.Popen(command, env=env, stdout=log_file, stderr=log_file)
    time.sleep(10)  # Give the server a few seconds to warm up
    print("vLLM server launched.")

# Inference Settings
VLLM_URL = "http://localhost:8000/v1/chat/completions"

SYSTEM_PROMPT = """You are an expert in financial transaction analysis. Your task is to generate clear, structured, and concise descriptions of user financial behavior based on given transaction data. Use data-driven insights and avoid speculation.
⚠️ Do **not** include introductory phrases like "Here's a financial behavior description for User X" or "Based on the provided data."  
⚠️ Start **directly** with the financial behavior insights.  
⚠️ Interpret numerical statistics **into meaningful behavioral patterns** rather than just restating them.  
⚠️ Highlight **spending habits, preferred transaction methods, risk patterns, and financial consistency**.  
⚠️ If applicable, discuss whether spending habits indicate **financial stability, risk-taking behavior, or specific lifestyle patterns**.  
⚠️ Keep responses **factual, structured, and data-driven** without making assumptions beyond the provided data.
"""

# ingle prompt inference
async def generate_one(prompt, user_id, client):
    """
    Sends one prompt to the vLLM server and returns the response.

    Args:
        prompt (str): Input prompt for the user.
        user_id (str): ID of the client/user.
        client (httpx.AsyncClient): Shared HTTP client session.

    Returns:
        dict: {"user_id": ..., "description": ...}
    """
    try:
        response = await client.post(VLLM_URL, json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.95
        })
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return {"user_id": user_id, "description": content}
    except Exception as e:
        return {"user_id": user_id, "description": f"[ERROR] {str(e)}"}

# Batched inference
async def generate_batch(batch):
    """
    Sends a batch of prompts to the vLLM server asynchronously.

    Args:
        batch (list): List of (user_id, prompt) tuples.

    Returns:
        list: List of generation results per user.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [generate_one(prompt, user_id, client) for user_id, prompt in batch]
        return await asyncio.gather(*tasks)

# Async generation pipelin
def async_generate_client_descriptions(df, user_col, time_col, amount_col, start_from_id=None,
                                 category_col=None, type_col=None,
                                 dataset_name="gender", output_path="client_descriptions.csv",
                                 batch_size=16, max_model_len=1500):
    """
    Processes the transaction dataset and generates behavioral descriptions using vLLM.

    Args:
        df (pd.DataFrame): Transaction dataset.
        user_col (str): Column name for user ID.
        time_col (str): Column name for timestamps.
        amount_col (str): Column name for transaction amounts.
        category_col (str, optional): Category feature name.
        type_col (str, optional): Type feature name.
        dataset_name (str): Dataset tag to customize logic.
        output_path (str): File to save the output.
        batch_size (int): Number of prompts per inference batch.
        max_model_len (int): Token limit for model context + generation.
    """

    # Launch the vLLM server
    launch_vllm_server(model_path=MODEL_NAME, port=8000, max_model_len=max_model_len)

    # Preprocess the data and generate prompts
    processed_df = preprocess_transactions(df, user_col, time_col, amount_col, category_col, type_col, dataset_name)
    if start_from_id is not None:
        processed_df = processed_df[processed_df[user_col] > start_from_id]
    prompts = generate_prompts(processed_df, user_col, dataset_name)

    # Create CSV output file if it doesn't exist
    if not Path(output_path).exists():
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "description"])
            writer.writeheader()

    # Inference in batches
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    for batch in tqdm(batches, total=len(batches), desc="Generating in parallel", ncols=90):
        results = asyncio.run(generate_batch(batch))
        with open(output_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "description"])
            for row in results:
                writer.writerow({"id": row["user_id"], "description": row["description"]})
