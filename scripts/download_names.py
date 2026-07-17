import pandas as pd
import os
import tempfile
from dotenv import load_dotenv
# Access Kaggle credentials from .env file
load_dotenv()
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')
import kaggle

# Download to a temp directory 
with tempfile.TemporaryDirectory() as tmpdir:
    kaggle.api.dataset_download_files(
        'kaggle/us-baby-names',
        path=tmpdir,
        unzip=True
    )
    df = pd.read_csv(os.path.join(tmpdir, 'NationalNames.csv'))
# tmpdir and its contents are deleted here, df lives in memory

# Only consider names from past 100 years
df = df[df['Year'] >= 1926]

# Take random sample of 1000 names from each gender
names = df.groupby('Gender').apply(lambda x: x.sample(1000, random_state=1234))
names.to_csv('../data/processed/random_names.csv', index=False)