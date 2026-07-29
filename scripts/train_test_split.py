
import pandas as pd 
import json
import regex as re
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# For reproducibility
SEED = 1234
np.random.seed(SEED)

# -------- UPLOADING THE DATA -------- 

print('Uploading the data...')

# Combined SafeDial + LMSYS dataset
X_multi = pd.read_csv('../data/processed/safedial_enriched_with_benign.csv')
X_multi['conversation'] = X_multi['conversation'].apply(json.loads)

# WildGuardMix already has train/ test split
X_train_sing = pd.read_parquet('../data/raw/wildguardmix/train/wildguard_train.parquet')
X_test_sing = pd.read_parquet('../data/raw/wildguardmix/test/wildguard_test.parquet')

# -------- FURTHER PRE-PROCESSING --------

print('Preprocessing...')

# Harmonize the conversation format for the single turn dataset with multi turn
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
X_train_sing = build_conversation(X_train_sing)
X_test_sing = build_conversation(X_test_sing)

# Convert conversations to string format
def conversation_to_text(conversation):
    """Flatten a list of turns into a single string, excluding role labels."""
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except:
            conversation = eval(conversation)
    return ' '.join(
        turn['content'] # exclude role entirely (trying to avoid issue of different turn distributions btwn SafeDial and LMSYS)
        for turn in conversation
        if isinstance(turn, dict)
    )

# Remove punctuation and stopwords
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
    
    stop_words = set(stopwords.words('english'))
    proc_text_no_stopwords = [word for word in proc_text.split() if word not in stop_words] # .split() for words
    proc_text_no_stopwords = ' '.join(proc_text_no_stopwords) # join back to string
    # proc_text_no_stopwords = proc_text_no_stopwords.replace('user', 'USER:')
    # proc_text_no_stopwords = proc_text_no_stopwords.replace('assistant', 'ASSISTANT:')

    return proc_text_no_stopwords

# Apply functions to preprocess all datasets
X_multi['conversation'] = X_multi['conversation'].apply(
    lambda x: preprocessor(conversation_to_text(x))
)
X_train_sing['conversation'] = X_train_sing['conversation'].apply(
    lambda x: preprocessor(conversation_to_text(x))
)
X_test_sing['conversation'] = X_test_sing['conversation'].apply(
    lambda x: preprocessor(conversation_to_text(x))
)

# Drop rows where preprocessing produced an empty string
X_multi = X_multi[X_multi['conversation'].str.strip() != '']
X_train_sing = X_train_sing[X_train_sing['conversation'].str.strip() != '']
X_test_sing = X_test_sing[X_test_sing['conversation'].str.strip() != '']

# -------- TRAIN, TEST, VAL SPLIT --------

print('Building train, test, and validation sets...')

# Train/ test/ validation for multi-turn

# Shuffle dataset
idx = X_multi.index.to_list()
np.random.shuffle(idx)
data_shuffled = X_multi.loc[idx].reset_index(drop=True)

X = data_shuffled[['conversation_id', 'conversation']]
y = data_shuffled[['harm']]

# Train + test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=SEED,
    stratify=y['harm'] # maintain class balance (dataset is already balanced 50-50)
)


# Train + val split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.25, 
    random_state=SEED,
    stratify=y_train['harm'] # maintain class balance (dataset is already balanced 50-50)
)

base_path = '../data/processed/'

print(f'Saving datasets to {base_path}')

# Save
X_train.to_csv(f'{base_path}multiturn_X_train.csv', index=False)
y_train.to_csv(f'{base_path}multiturn_Y_train.csv', index=False)
X_test.to_csv(f'{base_path}multiturn_X_test.csv', index=False)
y_test.to_csv(f'{base_path}multiturn_Y_test.csv', index=False)
X_val.to_csv(f'{base_path}multiturn_X_val.csv', index=False)
y_val.to_csv(f'{base_path}multiturn_Y_val.csv', index=False)

# Train/ test/ validation for single-turn

# Creating harmonized harm column based on prompt_harm_label column
X_train_sing['harm'] = np.where(X_train_sing['prompt_harm_label'] == 'harmful', True, False)
# Taking a sample of train set to match the multi-turn dataset
# Matching sample size to multi-turn dataset above
sample_size = len(X_train)+len(X_val)
# 50% harmful, 50% benign
X_train_sing_sampled = pd.concat([
    X_train_sing[X_train_sing['harm'] == True].sample(sample_size // 2 + sample_size % 2, random_state=SEED), # round up (uneven number)
    X_train_sing[X_train_sing['harm'] == False].sample(sample_size // 2, random_state=SEED) # round down (uneven number)
])
# Create conversation_id
X_train_sing_sampled['conversation_id'] = X_train_sing_sampled.reset_index().index

# Creating harmonized harm column based on prompt_harm_label column
X_test_sing['harm'] = np.where(X_test_sing['prompt_harm_label'] == 'harmful', True, False)
# Taking a sample of test set to match the multi-turn dataset
# Matching sample size to multi-turn dataset above
sample_size = len(X_test)
# 50% harmful, 50% benign
X_test_sing_sampled = pd.concat([
    X_test_sing[X_test_sing['harm'] == True].sample(sample_size // 2 + sample_size % 2, random_state=SEED), # round up (uneven number)
    X_test_sing[X_test_sing['harm'] == False].sample(sample_size // 2, random_state=SEED) # round down (uneven number)
])
# Avoiding overlap in ID with train dataset
X_test_sing_sampled['conversation_id'] = X_test_sing_sampled.index + len(X_train_sing_sampled)

# Shuffle dataset
idx = X_train_sing_sampled.index.to_list()
np.random.shuffle(idx)
data_shuffled = X_train_sing_sampled.loc[idx].reset_index(drop=True)

X_train = data_shuffled[['conversation_id', 'conversation']]
y_train = data_shuffled[['harm']]

# Train + val split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25, 
    random_state=SEED,
    stratify=y_train['harm'] # maintain class balance (dataset is already balanced 50-50)
)

# Test set aleady exists
X_test = X_test_sing_sampled[['conversation_id', 'conversation']]
y_test = X_test_sing_sampled[['harm']]

# Save
X_train.to_csv(f'{base_path}singleturn_X_train.csv', index=False)
y_train.to_csv(f'{base_path}singleturn_Y_train.csv', index=False)
X_test.to_csv(f'{base_path}singleturn_X_test.csv', index=False)
y_test.to_csv(f'{base_path}singleturn_Y_test.csv', index=False)
X_val.to_csv(f'{base_path}singleturn_X_val.csv', index=False)
y_val.to_csv(f'{base_path}singleturn_Y_val.csv', index=False)

print('Done!')






