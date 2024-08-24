import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


def prepare_data(csv_file, max_length=100):
    logging.info("Preparing data...")
    df = pd.read_csv(csv_file)
    df['text'] = df['clean_text'].fillna('').apply(
        lambda x: ' '.join(x.split()[:max_length]))

    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

    with open('train_data.txt', 'w', encoding='utf-8') as f:
        for text in train_df['text']:
            f.write(f"{text}\n")

    with open('eval_data.txt', 'w', encoding='utf-8') as f:
        for text in eval_df['text']:
            f.write(f"{text}\n")

    logging.info(
        f"Saved {len(train_df)} training samples and {len(eval_df)} evaluation samples."
    )
    return len(train_df)


def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )


def train_model(num_train_samples):
    logging.info("Starting model training...")

    # Load pre-trained model and tokenizer
    model_name = "distilgpt2"  # Using a smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Prepare the dataset
    train_dataset = load_dataset("train_data.txt", tokenizer)
    eval_dataset = load_dataset("eval_data.txt", tokenizer)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)

    # Calculate the number of training steps (10% of original)
    num_train_steps = max(int((num_train_samples / 4) * 0.1),
                          100)  # Ensure at least 100 steps

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=200,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=max(int(num_train_steps / 10),
                       10),  # Evaluate roughly 10 times during training
        save_steps=num_train_steps,  # Save only at the end
        warmup_steps=min(int(num_train_steps * 0.1),
                         100),  # 10% of steps for warmup, max 100
        logging_dir='./logs',
        logging_steps=max(int(num_train_steps / 20),
                          5),  # Log roughly 20 times during training
        logging_first_step=True,
        max_steps=num_train_steps,  # Set the total number of training steps
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    try:
        trainer.train()
        # Save both the model and tokenizer
        trainer.save_model("./mental_health_chatbot_model")
        tokenizer.save_pretrained("./mental_health_chatbot_model")
        logging.info("Model and tokenizer training completed and saved.")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        sys.exit(1)


def generate_response(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids,
                            max_length=100,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    csv_file = 'depression_dataset_reddit_cleaned.csv'  # Make sure this file is in your working directory

    if not os.path.exists(csv_file):
        logging.error(f"CSV file not found: {csv_file}")
        sys.exit(1)

    num_train_samples = prepare_data(csv_file)
    train_model(num_train_samples)

    # Load the trained model for inference
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "./mental_health_chatbot_model").to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "./mental_health_chatbot_model")

        # Example usage
        prompt = "I've been feeling really anxious lately..."
        response = generate_response(prompt, model, tokenizer)
        print(f"User: {prompt}")
        print(f"Chatbot: {response}")
    except Exception as e:
        logging.error(f"An error occurred during inference: {str(e)}")
        sys.exit(1)
