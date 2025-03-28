import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
import numpy as np
import csv
import os

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

    # Compute additional time-based features
    df["prev_time"] = df.groupby(user_col)[time_col].shift(1)

    if dataset_name.lower() in ["gender", "age group"]:
        df["time_diff"] = df[time_col] - df["prev_time"]
    else:
        df["time_diff"] = (df[time_col] - df["prev_time"]).dt.days

    avg_interval = df.groupby(user_col)["time_diff"].mean().reset_index().rename(columns={"time_diff": "avg_transaction_interval"})
    grouped = grouped.merge(avg_interval, on=user_col, how="left")

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
        mcc_counts = df.groupby([user_col, category_col]).size().reset_index(name="count")

        # Compute total transaction amounts per MCC using absolute values
        df["abs_amount"] = df[amount_col].abs()
        mcc_sums = df.groupby([user_col, category_col])["abs_amount"].sum().reset_index()

        top_mccs = mcc_counts.sort_values(by=["count"], ascending=False).groupby(user_col).head(3)

         # Assign rank to top MCCs
        top_mccs["rank"] = top_mccs.groupby(user_col)["count"].rank(method="first", ascending=False) - 1  # 0, 1, 2

        # Pivot to get mcc_0, mcc_1, mcc_2
        top_mcc_pivot = top_mccs.pivot(index=user_col, columns="rank", values=category_col).rename(
            columns={0: "mcc_0", 1: "mcc_1", 2: "mcc_2"}
        )

        # Merge back into grouped dataframe
        grouped = grouped.merge(top_mcc_pivot, on=user_col, how="left")

        # Compute the share of transactions from the most common MCC (by amount)
        top1_mcc = top_mccs[top_mccs["rank"] == 0][[user_col, category_col]].rename(columns={category_col: "top1_mcc"})

        # Merge with total MCC transaction amounts
        top1_mcc = top1_mcc.merge(mcc_sums, left_on=[user_col, "top1_mcc"], right_on=[user_col, category_col], how="left")

        # Fill NaN values in transaction amounts with 0
        top1_mcc["abs_amount"] = top1_mcc["abs_amount"].fillna(0)

        # Compute share of top MCC transactions relative to total user spending
        if dataset_name == 'gender':
            total_spent = (grouped["total_income"] + grouped["total_expenses"]) # Avoid division by zero
        else:
            total_spent = grouped["total_sum"]

        # Compute total transaction amount for top1 MCC per user
        top1_mcc_share_df = top1_mcc.groupby(user_col)["abs_amount"].sum().reset_index()

        # Merge with grouped DataFrame to ensure alignment
        grouped = grouped.merge(top1_mcc_share_df.rename(columns={"abs_amount": "top1_mcc_total"}), on=user_col, how="left")

        # Compute share of transactions for the top MCC category
        grouped["top1_mcc_share"] = grouped["top1_mcc_total"] / total_spent

        # Compute average amount per MCC using absolute values
        avg_mcc_amounts = df.groupby([user_col, category_col])["abs_amount"].mean().reset_index()

        # Merge average amounts with top MCCs to ensure correct mapping
        top_mccs = top_mccs.merge(avg_mcc_amounts, on=[user_col, category_col], how="left")

        # Pivot to get avg_amount_mcc_0, avg_amount_mcc_1, avg_amount_mcc_2
        avg_mcc_pivot = top_mccs.pivot(index=user_col, columns="rank", values="abs_amount").rename(
            columns={0: "avg_amount_mcc_0", 1: "avg_amount_mcc_1", 2: "avg_amount_mcc_2"}
        )
        # Merge into final grouped dataframe
        grouped = grouped.merge(avg_mcc_pivot, on=user_col, how="left")
    
    # Compute the fraction of days with transactions
    transaction_days = df.groupby(user_col)[time_col].nunique().reset_index(name="unique_transaction_days")
    grouped = grouped.merge(transaction_days, on=user_col, how="left")
    grouped["transaction_days_share"] = grouped["unique_transaction_days"] / grouped["transaction_period"]

    # Extract most common transaction type correctly
    if type_col:
        type_mode = df.groupby(user_col)[type_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
        grouped = grouped.merge(type_mode.rename("most_common_type"), on=user_col, how="left")

    return grouped

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
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating prompts", ncols=80):
        user_info = f"""
        User {row[user_col]}:
        - Number of transactions: {row['transaction_count']}
        - Active transaction period: {row['transaction_period']} days
        - Avg transactions per day: {row['avg_transaction_per_day']:.2f}
        - Avg transaction interval: {row['avg_transaction_interval']:.2f} days
        - Top MCCs: {row.get('mcc_0', 'N/A')}, {row.get('mcc_1', 'N/A')}, {row.get('mcc_2', 'N/A')}
        - Share of transactions in Top 1 MCC: {row.get('top1_mcc_share', 0):.2f}
        - Avg amount for Top 1 MCC: {row.get('avg_amount_mcc_0', 0):.2f} RUB
        - Avg amount for Top 2 MCC: {row.get('avg_amount_mcc_1', 0):.2f} RUB
        - Avg amount for Top 3 MCC: {row.get('avg_amount_mcc_2', 0):.2f} RUB
        - Share of days with transactions: {row['transaction_days_share']:.2f}
        """

        if dataset_name.lower() == "gender":
            user_info += f"- Total income: {row['total_income']:.2f} RUB\n"
            user_info += f"- Total expenses: {row['total_expenses']:.2f} RUB\n"
            user_info += f"- Average outgoing transaction amount: {row['avg_spent']:.2f} RUB\n"
            user_info += f"- Average incoming transaction amount: {row['avg_received']:.2f} RUB\n"
        else:
            user_info += f"- Total sum of transactions: {row['total_sum']:.2f} RUB\n"
            user_info += f"- Average transaction amount: {row['avg_spent']:.2f} RUB\n"
        
        if "most_common_type" in row:
            user_info += f"- Most common transaction type: {row['most_common_type']}\n"

        example_description = """
        "This user exhibits **infrequent but high-value transactions**, suggesting either planned purchases or a preference for bulk spending rather than daily transactions. Their **long average transaction interval** and **low transaction frequency per day** indicate that they do not engage in routine or habitual spending.
        The majority of transactions are concentrated in the **electronics sector**, with the top MCC category accounting for a significant portion of total spending. This suggests a preference for high-cost items or infrequent but large purchases rather than day-to-day expenses like groceries or utilities.
        The **low share of days with transactions** further confirms that financial activity is not evenly distributed, which may indicate a budgeting strategy where the user makes **large planned purchases and avoids daily expenditures**.
        If the user has both **income and expenses recorded**, a high discrepancy between income and expenses could indicate **either a reliance on stored financial reserves or irregular income patterns**. Conversely, if income and expenses are relatively balanced, the user likely **follows a structured financial approach with periodic spending bursts**.
        While there are no apparent high-risk spending patterns, the concentration in a few transaction categories suggests that **this user may not diversify their financial activities**, which could lead to higher exposure to financial shocks in those specific spending areas."
        """
        
        prompt = f"""
        Below is a summary of a user's transaction history:
        {user_info}
        
        ### Instructions:
        - **Analyze behavioral patterns** based on the statistics.
        - **Identify transaction regularity** (e.g., consistent spending, seasonal trends, impulse purchases).
        - **Determine reliance on certain categories** (e.g., is spending concentrated in one or two areas, or is it well-diversified?).
        - **Assess potential risk factors** (e.g., irregular spending, high single-category reliance, inconsistent income flow).
        - **Identify financial planning traits** (e.g., stable spending habits vs. erratic transactions).
        - **Write in a structured and engaging way** while staying factual.

        ### Example Response:
        {example_description}
        """

        prompts.append((row[user_col], prompt))
    
    return prompts
