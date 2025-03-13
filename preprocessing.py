import pandas as pd
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from datetime import datetime

def parse_timestamp(timestamp, dataset_name):
    """
    Parses transaction timestamps based on dataset-specific formats.

    Args:
        timestamp (str, int, or pd.Timestamp): The timestamp value from the dataset.
        dataset_name (str): The dataset name to determine the parsing method.

    Returns:
        pd.Timestamp or int: A standardized timestamp format.
    """
    if isinstance(timestamp, pd.Timestamp):
        return timestamp  # Already a valid timestamp, return as is
    
    if isinstance(timestamp, int):
        return timestamp  # Age dataset uses integers, no conversion needed
    
    if isinstance(timestamp, str):
        if dataset_name.lower() == "gender":
            # Extract the day from '0 10:23:26' → converts "0" to int
            return int(timestamp.split(" ")[0])
        
        elif dataset_name.lower() == "churn":
            # Converts '18APR17:00:00:00' → datetime (18th April 2017)
            try:
                return pd.to_datetime(timestamp[:7], format="%d%b%y", errors='coerce')
            except ValueError:
                return pd.NaT  # Return Not-a-Time if parsing fails

        else:
            # Default case: Convert any other string-based timestamp to datetime
            return pd.to_datetime(timestamp, errors='coerce')

    # If it's an unrecognized format, return NaT
    return pd.NaT

# Function to preprocess transaction data and extract key features
def preprocess_transactions(df, user_col, time_col, amount_col, category_col=None, type_col=None, dataset_name=""):
    """
    Groups transactions by user and calculates key statistics for each client.

    Args:
        df (pd.DataFrame): The input transaction dataset.
        user_col (str): The column name for the user ID.
        time_col (str): The column name for transaction timestamps.
        amount_col (str): The column name for transaction amounts.
        category_col (str, optional): The column name for transaction categories (MCC, small_group, etc.).
        type_col (str, optional): The column name for transaction types (POS, ATM withdrawal, transfer, etc.).
        dataset_name (str): The dataset name to adjust preprocessing logic.

    Returns:
        pd.DataFrame: A grouped dataset with aggregated transaction features per user.
    """
    # Convert timestamps properly
    df[time_col] = df[time_col].apply(lambda x: parse_timestamp(x, dataset_name))

    # Sort transactions chronologically per user
    df = df.sort_values(by=[user_col, time_col])

    # Base aggregations for all datasets
    grouped = df.groupby(user_col).agg(
        transaction_count=(amount_col, "count"),
        first_transaction=(time_col, "min"),
        last_transaction=(time_col, "max")
    ).reset_index()

    if dataset_name.lower() in ["gender", "age group"]:
        grouped["transaction_period"] = grouped["last_transaction"] - grouped["first_transaction"]
    else:
        grouped["transaction_period"] = (grouped["last_transaction"] - grouped["first_transaction"]).dt.days

    # Prevent division by zero
    grouped["transaction_period"] = grouped["transaction_period"].replace(0, 1)
    grouped["avg_transaction_per_day"] = grouped["transaction_count"] / grouped["transaction_period"]

    # Special processing for Gender Prediction dataset
    if dataset_name.lower() == "gender":
        income_df = df[df[amount_col] > 0].groupby(user_col)[amount_col].agg(["sum", "mean"]).rename(columns={"sum": "total_income", "mean": "avg_received"})
        expenses_df = df[df[amount_col] < 0].groupby(user_col)[amount_col].agg(["sum", "mean"]).rename(columns={"sum": "total_expenses", "mean": "avg_spent"})
        
        # Convert negative values in expenses to positive
        expenses_df["total_expenses"] = expenses_df["total_expenses"].abs()
        expenses_df["avg_spent"] = expenses_df["avg_spent"].abs()
        
        # Merge income & expenses into main dataframe
        grouped = grouped.merge(income_df, on=user_col, how="left").merge(expenses_df, on=user_col, how="left")

    else:
        total_sum_df = df.groupby(user_col, as_index=False).agg({amount_col: "sum"}).rename(columns={amount_col: "total_sum"})
        avg_spent_df = df.groupby(user_col, as_index=False).agg({amount_col: "mean"}).rename(columns={amount_col: "avg_spent"})
        grouped = grouped.merge(total_sum_df, on=user_col, how="left").merge(avg_spent_df, on=user_col, how="left")

    # Ensure missing values are filled with 0
    grouped = grouped.fillna(0)

    # Extract most common transaction category correctly
    if category_col:
        category_mode = df.groupby(user_col)[category_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
        grouped = grouped.merge(category_mode.rename("most_common_category"), on=user_col, how="left")

    # Extract most common transaction type correctly
    if type_col:
        type_mode = df.groupby(user_col)[type_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
        grouped = grouped.merge(type_mode.rename("most_common_type"), on=user_col, how="left")

    return grouped

# Function to generate adaptive prompts based on dataset characteristics
def generate_prompts(df, user_col, dataset_name):
    """
    Creates structured prompts for each user based on their transaction history.

    Args:
        df (pd.DataFrame): The processed dataset with aggregated features.
        user_col (str): The column name for the user ID.
        dataset_name (str): The dataset name to adjust prompt formatting.

    Returns:
        list of tuples: (user_id, prompt)
    """
    prompts = []
    
    for _, row in df.iterrows():
        user_info = f"""
        User {row[user_col]}:
        - Number of transactions: {row['transaction_count']}
        - Active transaction period: {row['transaction_period']} days
        - Avg transactions per day: {row['avg_transaction_per_day']:.2f}
        """

        # Handle dataset-specific differences
        if dataset_name.lower() == "gender":
            user_info += f"- Total income: {row['total_income']:.2f} RUB\n"
            user_info += f"- Total expenses: {row['total_expenses']:.2f} RUB\n"
            user_info += f"- Average outgoing transaction amount: {row['avg_spent']:.2f} RUB\n"
            user_info += f"- Average incoming transaction amount: {row['avg_received']:.2f} RUB\n"
        else:
            user_info += f"- Total sum of transactions: {row['total_sum']:.2f} RUB\n"
            user_info += f"- Average transaction amount: {row['avg_spent']:.2f} RUB\n"

        # Add most common transaction category & type if available
        if "most_common_category" in row:
            user_info += f"- Most frequent spending category: {row['most_common_category']}\n"
        if "most_common_type" in row:
            user_info += f"- Most common transaction type: {row['most_common_type']}\n"

        example_description = """
        Example: 
        "This user primarily spends on retail purchases, with grocery shopping being the dominant category. 
        They frequently withdraw cash from ATMs, indicating a preference for cash transactions. 
        Their spending is consistent, with occasional online purchases such as subscriptions."
        """
        
        prompt = f"""
        Below is a summary of a user's transaction history:
        {user_info}
        
        Based on this information, generate a concise yet insightful financial behavior description.
        Your response should be clear, structured, and informative.
        {example_description}
        """
        
        prompts.append((row[user_col], prompt))
    
    return prompts

def query_llm(prompt, hf_token):
    """
    Sends the prompt to the LLM and retrieves a generated response.

    Args:
        prompt (str): The prompt describing a user's transaction behavior.

    Returns:
        str: Generated financial behavior summary from the model.
    """

    system_message = """You are an expert in financial transaction analysis. Your task is to generate clear, structured, and concise descriptions of user financial behavior based on given transaction data. Use data-driven insights and avoid speculation.
    ⚠️ Do **not** include introductory phrases like "Here's a financial behavior description for User X" or "Based on the provided data."  
    ⚠️ Start **directly** with the financial behavior insights.  
    ⚠️ Keep responses factual, structured, and data-driven. 
    """

    # Prepare the input message in the required format
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
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
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

# Main function: processes dataset and generates user descriptions
def generate_user_descriptions(df, user_col, time_col, amount_col, category_col=None, type_col=None, dataset_name="gender"):
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

    Returns:
        pd.DataFrame: A DataFrame with user descriptions generated by the LLM.
    """
    processed_df = preprocess_transactions(df, user_col, time_col, amount_col, category_col, type_col, dataset_name)
    prompts = generate_prompts(processed_df, user_col)

    descriptions = []

    # Initialize the Gemma 3 model pipeline
    model_id = "google/gemma-3-12b-it"
    hf_token = "YOUR_TOKEN"

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", token=hf_token).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    for user_id, prompt in prompts:
        llm_response = query_llm(prompt)
        descriptions.append({"user_id": user_id, "description": llm_response})

    return pd.DataFrame(descriptions)
