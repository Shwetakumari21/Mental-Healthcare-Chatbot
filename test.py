import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


def save_and_load_test(model_name="distilgpt2", save_path="./test_model"):
	try:
		# Load pre-trained model and tokenizer
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

		# Save model and tokenizer
		logging.info(f"Saving model and tokenizer to {save_path}")
		model.save_pretrained(save_path)
		tokenizer.save_pretrained(save_path)

		# Try to load the saved model and tokenizer
		logging.info(
		    f"Attempting to load model and tokenizer from {save_path}")
		loaded_model = AutoModelForCausalLM.from_pretrained(save_path).to(
		    device)
		loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)

		logging.info("Successfully loaded the saved model and tokenizer")

		# Test the loaded model
		test_input = "i hate myself so much"
		input_ids = loaded_tokenizer.encode(test_input,
		                                    return_tensors='pt').to(device)
		output = loaded_model.generate(input_ids, max_length=50)
		generated_text = loaded_tokenizer.decode(output[0],
		                                         skip_special_tokens=True)

		logging.info(f"Test input: {test_input}")
		logging.info(f"Generated output: {generated_text}")

	except Exception as e:
		logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
	save_and_load_test()

	# List contents of the save directory
	save_path = "./test_model"
	if os.path.exists(save_path):
		logging.info(f"Contents of {save_path}:")
		for file in os.listdir(save_path):
			logging.info(f"- {file}")
	else:
		logging.error(f"{save_path} does not exist")
