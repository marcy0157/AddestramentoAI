Description:
This script performs text classification model training using DistilBERT with the Hugging Face Transformers library. 
The model classifies combined sentences from two input columns (Title and Description) into two labels (Relevant and Not Relevant).

Prerequisites:
1. Python: Version 3.8 or higher.
2. GPU (optional): For faster training, a GPU with CUDA support is recommended.
3. Required libraries: Install the following libraries using the command:
   pip install datasets transformers scikit-learn torch

Dataset:
The dataset must be a CSV file with the following columns:
- Title: The title of the content.
- Description: A brief description of the content.
- Relevance: The target label with values "Relevant" or "Not Relevant".

Sample dataset (dataset.csv):
| Title          | Description          | Relevance     |
|-----------------|----------------------|---------------|
| Example title 1 | Example description | Relevant      |
| Example title 2 | Another description | Not Relevant  |

Usage:
1. Prepare the dataset:
   Ensure the CSV file is in the same directory as the script or specify the correct path in the code:
   dataset = load_dataset('csv', data_files='dataset.csv')

2. Run the script:
   python training_script.py

3. Monitor the process:
   During training, progress and evaluation metrics will be displayed.

4. Results:
   - Model checkpoints will be saved in the ./results directory.
   - Logs will be saved in the ./logs directory.

Configurable Parameters:
- Batch size: Modify the per_device_train_batch_size and per_device_eval_batch_size parameters in TrainingArguments.
- Learning rate: Customize the learning_rate parameter in TrainingArguments.
- Number of epochs: Adjust the num_train_epochs value in TrainingArguments.
- Checkpoints: Configure save_steps and save_total_limit to manage checkpoint frequency.

Hardware Requirements:
- GPU: If available, the script will automatically use CUDA. Otherwise, it will run on the CPU.

Script Customization:
The script can be easily modified to:
- Add new columns to the dataset.
- Change the base model by replacing "distilbert-base-uncased" with another Hugging Face model.
