import pandas as pd
import numpy as np
import glob
import json
import re
from sklearn.model_selection import train_test_split

# For reproducibility
SEED = 1234
np.random.seed(SEED)

#### SINGLE-TURN DATASET PREPROCESSING ####

print('Preprocessing single-turn datasets...')

# WildGuardMix is a dataset of single-turn human-LLM interactions (both benign and harmful)
# WildGuardMix already has train/ test split

# Uploading
singleturn_train = pd.read_parquet('../data/raw/wildguardmix/train/wildguard_train.parquet')
singleturn_test = pd.read_parquet('../data/raw/wildguardmix/test/wildguard_test.parquet')

# Remove duplicate prompts
singleturn_train = singleturn_train.drop_duplicates(subset=['prompt'])
singleturn_test = singleturn_test.drop_duplicates(subset=['prompt'])

# Creating harm column based on prompt_harm_label and response_harm_label
# Flag for whether or not any part of the conversation is harmful
singleturn_train['harm'] = np.where((singleturn_train['prompt_harm_label'] == 'harmful') | (singleturn_train['response_harm_label'] == 'harmful'), 1, 0)
singleturn_test['harm'] = np.where((singleturn_test['prompt_harm_label'] == 'harmful') | (singleturn_test['response_harm_label'] == 'harmful'), 1, 0)

# Harmonize the conversation format for the single-turn dataset with multi-turn
# Combines prompt by user and response by LLM into one conversation column
def build_conversation(df):
    # Remove any rows with missing prompts
    df = df.dropna(subset=['prompt'])
    df['conversation'] = df.apply(
        lambda row: [
            {'role': 'user', 'content': row['prompt']},
            # If response is blank, just convert it to an empty string
            {'role': 'assistant', 'content': row['response'] if pd.notna(row['response']) else ''}
        ],
        axis=1
    )
    return df

# Apply harmonization
singleturn_train = build_conversation(singleturn_train)
singleturn_test = build_conversation(singleturn_test)

# Create a column that holds the length of each convo
def conversation_length(conversation):
    """Handles numpy arrays, lists of dicts, and plain strings"""
    if isinstance(conversation, (list, np.ndarray)):
        return sum(
            len(msg.get('content', '').split()) 
            for msg in conversation 
            if isinstance(msg.get('content'), str)
        )
    elif isinstance(conversation, str):
        return len(conversation.split())
    return 0

# Compute lengths 
singleturn_train['conv_length'] = singleturn_train['conversation'].apply(conversation_length)
singleturn_test['conv_length'] = singleturn_test['conversation'].apply(conversation_length)

#### MULTI-TURN DATASET PREPROCESSING ####

print('Preprocessing multi-turn dataset...')

# LMSYS is a dataset or real world human-LLM interactions (single- and multi-turn)
# Posts flagged by OpenAI moderator for harm when applicable
# As a generalizability test, the model trained on WildGuardMix's single-turn conversations will be tested on longer conversations from a novel dataset

# Compile all LMSYS files
file_paths = glob.glob('../data/raw/lmsys-chat/data/*.parquet')
dfs = [pd.read_parquet(f, engine='pyarrow') for f in sorted(file_paths)]
lmsys = pd.concat(dfs).reset_index().drop(columns=['index'])

# Just English
lmsys_proc = lmsys[(lmsys['language'] == 'English')]

# Only considering multi-turn conversations with at least 3 turns
# Ensure increased complexity relative to single turn dataset for generalizability task
lmsys_proc = lmsys_proc[lmsys_proc['turn'] >= 3]

# Did OpenAI moderation produce any flags at any point in the convo?
lmsys_proc['any_flagged'] = lmsys_proc['openai_moderation'].apply(
    lambda turns: any(turn['flagged'] for turn in turns)
)
# Create harm column
lmsys_proc['harm'] = np.where(lmsys_proc['any_flagged'] == True, 1, 0)

# What categories was it flagged for?
def flagged_categories(turns):
    cats = set()
    for turn in turns:
        if turn['flagged']:
            cats.update(cat for cat, flagged in turn['categories'].items() if flagged)
    return list(cats)

# Find flagged categories
lmsys_proc['flagged_categories'] = lmsys_proc['openai_moderation'].apply(flagged_categories)

# Saving original conversation in this column
lmsys_proc.rename(columns={'conversation':'conversation_orig'},inplace=True)

# Apply standard format for conversations 
# Matches the format used to build a conversation dict in the single-turn dataset
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

# Apply formatting
lmsys_proc['conversation'] = lmsys_proc['conversation_orig'].apply(normalize_conversation)

# LMSYS redacts any names and replaces them with [NAME_X]
# Code below replaces [NAME_X] with random names from a name dataset

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
lmsys_proc['conversation'] = lmsys_proc.apply(
            lambda x: replace_names(x['conversation'], name_prob_dict) if x['redacted'] else x['conversation'],
            axis=1
        )

# Compute lengths 
lmsys_proc['conv_length'] = lmsys_proc['conversation'].apply(conversation_length)

# Excluding any conversations with word counts above 6,000 
# BERT model has context window of 8,000 tokens
# Conversion is ~1.3 tokens per word
lmsys_proc = lmsys_proc[lmsys_proc['conv_length'] < 6000]

# Taking a sample of the multiturn dataset 
# Will be used to test generalizability of model trained on single-turn conversations
# Split 50-50 harmful-unharmful

sample_size = 2000

multiturn = pd.concat([
    lmsys_proc[lmsys_proc['harm'] == True].sample(n=int(sample_size/2), random_state=SEED), 
    lmsys_proc[lmsys_proc['harm'] == False].sample(n=int(sample_size/2), random_state=SEED) 
])

### PREPROCESSING APPLIED TO BOTH DATASETS ###

print('Final preprocessing steps applied to all datasets...')

# Convert conversations to string format
def conversation_to_text(conversation):
    """Flatten a list of turns into a single string, preserving role labels."""
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except:
            conversation = eval(conversation)
    return ' '.join(
        f"__ROLE_{turn['role'].upper()}__ {turn['content']}"
        for turn in conversation
        if isinstance(turn, dict)
    )

# Clean strings
def preprocessor(conversation):
    if isinstance(conversation, list):
        text = conversation_to_text(conversation)
    else:
        text = conversation
    proc_text = text.replace('\n', ' ')
    proc_text = re.sub('<[^>]*>', '', proc_text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', proc_text)
    proc_text = (re.sub('[\W]+', ' ', proc_text.lower()) +
                 ' '.join(emoticons).replace('-', ''))

    # # Restore role labels (placeholders survived punctuation stripping since they're alphanumeric + _)
    # proc_text = proc_text.replace('__role_user__', 'USER:')
    # proc_text = proc_text.replace('__role_assistant__', 'ASSISTANT:')

    return proc_text

# Apply functions to preprocess all datasets

# Convert all entries to strings
# Do not apply further preprocessing to strings (for BERT model)
multiturn['conversation'] = multiturn['conversation'].apply(
    lambda x: conversation_to_text(x)
)
singleturn_train['conversation'] = singleturn_train['conversation'].apply(
    lambda x: conversation_to_text(x)
)
singleturn_test['conversation'] = singleturn_test['conversation'].apply(
    lambda x: conversation_to_text(x)
)

# Convert all entries to strings
# Apply further preprocessing to strings (for baseline and FFNN models)
multiturn['conversation_proc'] = multiturn['conversation'].apply(
    lambda x: preprocessor(x)
)
singleturn_train['conversation_proc'] = singleturn_train['conversation'].apply(
    lambda x: preprocessor(x)
)
singleturn_test['conversation_proc'] = singleturn_test['conversation'].apply(
    lambda x: preprocessor(x)
)

# Prepare to download full sample datasets (all columns) 
# Full datasets will be useful for EDA
# Polished datasets (select/ harmonized columns) generated below will be used for actual analysis 

# Serialize cleanly
multiturn['conversation'] = multiturn['conversation'].apply(json.dumps)
multiturn['flagged_categories'] = multiturn['flagged_categories'].apply(json.dumps)

# Save full multi-turn sample to path
path = '../data/processed/lmsys_multiturn_sample_full.csv'
multiturn.to_csv(path, index=False)
print(f'Saving full multi-turn sample (for EDA) to {path}')

# Save full single-turn datasets to paths
path = '../data/processed/wildguardmix_singleturn_train_sample_full.csv'
singleturn_train.to_csv(path, index=False)
print(f'Saving full single-turn train sample (for EDA) to {path}')
path = '../data/processed/wildguardmix_singleturn_test_sample_full.csv'
multiturn.to_csv(path, index=False)
print(f'Saving full single-turn test sample (for EDA) to {path}')

### TRAIN, TEST, VAL SPLIT ###

# Polished train/ test/ validation for single-turn

print(f'Building polished single-turn train, test, and validation sets...')

# Create conversation_id
singleturn_train['conversation_id'] = singleturn_train.index
# Avoiding overlap in ID with train dataset
singleturn_test['conversation_id'] = singleturn_test.index + len(singleturn_train)

# Shuffle dataset
idx = singleturn_train.index.to_list()
np.random.shuffle(idx)
data_shuffled = singleturn_train.loc[idx].reset_index(drop=True)

X_train = data_shuffled[['conversation_id', 'conversation', 'conversation_proc']]
y_train = data_shuffled[['conversation_id', 'harm']]

# Train + val split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.20, 
    random_state=SEED,
    stratify=y_train['harm'] # maintain class balance 
)

# Test set aleady exists

# Shuffle dataset
idx = singleturn_test.index.to_list()
np.random.shuffle(idx)
data_shuffled = singleturn_test.loc[idx].reset_index(drop=True)

X_test = data_shuffled[['conversation_id', 'conversation', 'conversation_proc']]
y_test = data_shuffled[['conversation_id', 'harm']]

base_path = '../data/processed/'

print(f'Saving polished single-turn datasets (for analysis) to {base_path} folder')

# Save
X_train.to_csv(f'{base_path}singleturn_X_train.csv', index=False)
y_train.to_csv(f'{base_path}singleturn_Y_train.csv', index=False)
X_test.to_csv(f'{base_path}singleturn_X_test.csv', index=False)
y_test.to_csv(f'{base_path}singleturn_Y_test.csv', index=False)
X_val.to_csv(f'{base_path}singleturn_X_val.csv', index=False)
y_val.to_csv(f'{base_path}singleturn_Y_val.csv', index=False)

# Now building polished multi-turn extension test set 

print(f'Building polished multi-turn dataset for additional testing...')

# Shuffle dataset
idx = multiturn.index.to_list()
np.random.shuffle(idx)
data_shuffled = multiturn.loc[idx].reset_index(drop=True)

X = data_shuffled[['conversation_id', 'conversation', 'conversation_proc']]
y = data_shuffled[['conversation_id', 'harm']]

print(f'Saving polished multi-turn test datasets (for analysis) to {base_path} folder')

# Save
X.to_csv(f'{base_path}multiturn_X_test.csv', index=False)
y.to_csv(f'{base_path}multiturn_Y_test.csv', index=False)

print('Done!')