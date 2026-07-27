import pandas as pd
import glob
import json
import numpy as np
import re

# -------- UPLOADING THE DATA -------- 

print('Uploading the data...')

# Uploading SafeDial (only harmful mutli-prompt examples)

# Combine all .jsonl files into single dataset
file_paths = glob.glob('../data/raw/safedialbench/data/by_task/english/*.jsonl')

dfs = []
for path in file_paths:
    rows = []
    with open(path, 'r') as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df['task'] = path.split('/')[-1].replace('.jsonl', '') 
    dfs.append(df)

safedial = pd.concat(dfs, ignore_index=True)

# Uploading LMSYS dataset (real world conversations between humans and LLM models)
# Will use posts NOT flagged by OpenAI moderation as "benign" multi-prompt conversations
# Enriches data provided by the SafeDial dataset, which only includes examples of harmful multi-prompt conversations

# Compile all lmsys files
file_paths = glob.glob('../data/raw/lmsys-chat/data/*.parquet')
dfs = [pd.read_parquet(f, engine='pyarrow') for f in sorted(file_paths)]
lmsys = pd.concat(dfs).reset_index().drop(columns=['index'])

# -------- PRE-PROCESSING THE DATA -------- 

print('Pre-processing the data...')

# SafeDial processing 

safedial_proc = safedial.copy()

# Adding turn column 
safedial_proc['turn'] = safedial_proc['history'].apply(lambda x: len(x))

# LMSYS processsing 

# Just English
lmsys_proc = lmsys[(lmsys['language'] == 'English')]
# Only considering multi-turn conversations
lmsys_proc = lmsys_proc[lmsys_proc['turn'] > 1]

# Did OpenAI moderation produce any flags at any point in the convo?
lmsys_proc['any_flagged'] = lmsys_proc['openai_moderation'].apply(
    lambda turns: any(turn['flagged'] for turn in turns)
)

# Only considering posts not flagged by OpenAI moderation (will act as benign counterparts to SafeDial's harmful examples)
lmsys_proc = lmsys_proc[lmsys_proc['any_flagged'] == False]

# What categories was it flagged for?
def flagged_categories(turns):
    cats = set()
    for turn in turns:
        if turn['flagged']:
            cats.update(cat for cat, flagged in turn['categories'].items() if flagged)
    return list(cats)

# Find flagged categories
lmsys_proc['flagged_categories'] = lmsys_proc['openai_moderation'].apply(flagged_categories)

# LMSYS redacts any names and replaces them with [NAME_X]
# Code below replaces [NAME_X] with random names from a name dataset
# This will avoid the model associating [NAME_X] with benign conversations (since only lmsys has this element)

# Names dataset
names = pd.read_csv('../data/processed/random_names.csv')
name_prob_dict = dict(zip(names['Name'], names['Count']))

def replace_names(entry, name_prob_dict):
    names = list(name_prob_dict.keys())
    counts = np.array(list(name_prob_dict.values()))
    probs = counts / counts.sum()

    # Find all unique NAME_X tokens across all messages first
    all_tokens = set()
    for msg in entry:
        all_tokens.update(re.findall(r'NAME_\d+', msg['content']))

    # Assign each token a fixed name
    token_map = {token: np.random.choice(names, p=probs) for token in all_tokens}

    result = []
    for msg in entry:
        new_msg = msg.copy()
        new_msg['content'] = re.sub(r'NAME_\d+', lambda m: token_map[m.group()], msg['content'])
        result.append(new_msg)
    return result

# Replace redactions with names
lmsys_proc['conversation_enriched'] = lmsys_proc.apply(
            lambda x: replace_names(x['conversation'], name_prob_dict) if x['redacted'] else x['conversation'],
            axis=1
        )

# Processing that applies to both datasets 

# Normalize conversation formats between the two datasets
def normalize_conversation(x):
    """Convert any conversation format to a unified {'role': ..., 'content': ...} schema."""
    # Handle ndarray from parquet
    if isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, str):
        try:
            x = json.loads(x)
        except:
            x = eval(x)
    
    normalized = []
    for turn in x:
        # ndarray elements may also need converting
        if isinstance(turn, np.ndarray):
            turn = turn.tolist()
        
        if 'role' in turn and 'content' in turn:
            normalized.append({'role': turn['role'], 'content': turn['content']})
        elif 'user' in turn and 'bot' in turn:
            normalized.append({'role': 'user', 'content': turn['user']})
            normalized.append({'role': 'assistant', 'content': turn['bot']})
        else:
            normalized.append(dict(turn)) # convert any remaining ndarray-backed dicts
    
    return normalized

# Normalize
safedial_proc['history'] = safedial_proc['history'].apply(normalize_conversation)
lmsys_proc['conversation_enriched'] = lmsys_proc['conversation_enriched'].apply(normalize_conversation)

# Create a column that holds the length of each convo
def conversation_length(conversation):
    """Handles numpy arrays, lists of dicts, and plain strings"""
    if isinstance(conversation, (list, np.ndarray)):
        return sum(
            len(msg.get('content', '')) 
            for msg in conversation 
            if isinstance(msg.get('content'), str)
        )
    elif isinstance(conversation, str):
        return len(conversation)
    return 0

# Compute lengths 
safedial_proc['conv_length'] = safedial_proc['history'].apply(conversation_length)
lmsys_proc['conv_length'] = lmsys_proc['conversation_enriched'].apply(conversation_length)

# Saving for EDA purposes
safedial_proc.to_csv('../data/processed/safedial_processed.csv', index=False)
lmsys_proc.to_csv('../data/processed/lmsys_processed.csv', index=False)

# -------- BUILDING COMBINED MULTI-TURN DATASET -------- 

print('Building combined multi-turn dataset...')

# Will stratify benign sample from LMSYS by conversation length

# Define bins based on harmful's distribution
n_bins = 20
_, bin_edges = np.histogram(safedial_proc['conv_length'], bins=n_bins)

# Expand left edge slightly to include minimum value
bin_edges[0] = bin_edges[0] - 0.001

harmful_binned = pd.cut(safedial_proc['conv_length'], bins=bin_edges)
harmful_bin_counts = harmful_binned.value_counts().sort_index()

benign_binned = pd.cut(lmsys_proc['conv_length'], bins=bin_edges)
benign_bin_counts = benign_binned.value_counts().sort_index()

comparison = pd.DataFrame({
    'harmful_needed': harmful_bin_counts,
    'benign_available': benign_bin_counts
}).fillna(0).astype(int)

SEED = 1234

# Building benign sample, stratified by 20 conversation length bins
samples = []
for interval, row in comparison.iterrows():
    n_needed = row['harmful_needed']
    bin_data = lmsys_proc[
        (lmsys_proc['conv_length'] > interval.left) &
        (lmsys_proc['conv_length'] <= interval.right)
    ]
    samples.append(bin_data.sample(n_needed, random_state=SEED))

benign_length_matched = pd.concat(samples)

# Merge SafeDial (harmful) and stratified sample from LMSYS (benign)

# Stratified sample from lmsys = benign
benign_length_matched['harm'] = False
# Original safedial data = harmful
safedial_proc['harm'] = True

merged_data = pd.concat([
    safedial_proc[['id', 'history', 'harm']].rename(columns={'id': 'conversation_id', 'history': 'conversation'}),
    benign_length_matched.drop(columns=['conversation']).rename(columns={'conversation_enriched':'conversation'})[['conversation_id', 'conversation', 'harm']]
]).reset_index().drop(columns=['index'])

# Serialize cleanly
merged_data['conversation'] = merged_data['conversation'].apply(json.dumps)

path = '../data/processed/safedial_enriched_with_benign.csv'
print(f'Saving to {path}')

# Save to path
merged_data.to_csv(path, index=False)

print('Done!')
