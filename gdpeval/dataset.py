from datasets import load_dataset

ds = load_dataset("openai/gdpval")

# --- Explore the dataset ---

# View available splits (train, test, validation, etc.)
print(ds)

# View column names
print(ds["train"].column_names)

# View first few rows as a table
print(ds["train"][:5])

# View a single example (as dict)
print(ds["train"][0])

# --- Convert to pandas for easier exploration ---
df = ds["train"].to_pandas()
print(df.head())
print(df.info())

# --- Save to JSON ---
# Option 1: JSONL format (one JSON object per line, no commas)
ds["train"].to_json("gdp_eval_train.jsonl")

# Option 2: Proper JSON array with commas (use pandas)
df.to_json("gdp_eval_train.json", orient="records", indent=2)
