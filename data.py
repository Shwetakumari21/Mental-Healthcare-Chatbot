import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
# Make sure you've downloaded the CSV file from Kaggle and placed it in the correct directory
df = pd.read_csv('depression_dataset_reddit_cleaned.csv')

# Preprocess the data
# Here we're just using the 'clean_text' column, but you might want to include other relevant columns
df['text'] = df['clean_text'].fillna('')

# Split into train and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)


# Function to save data in a format suitable for the Transformers library
def save_data_for_transformers(dataframe, filename):
	with open(filename, 'w', encoding='utf-8') as f:
		for text in dataframe['text']:
			f.write(f"{text}\n")


# Save train and eval datasets
save_data_for_transformers(train_df, 'train_data.txt')
save_data_for_transformers(eval_df, 'eval_data.txt')

print(
    f"Saved {len(train_df)} training samples and {len(eval_df)} evaluation samples."
)
