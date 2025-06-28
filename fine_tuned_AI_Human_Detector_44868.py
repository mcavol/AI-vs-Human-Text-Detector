# requirements numpy<2.0, transformers, datasets, pandas

# Load the Dataset with 44868 examples
import pandas as pd
from datasets import Dataset

# URL to CSV file
file_path = 'https://drive.usercontent.google.com/download?id=16jly0gQIVd1j3k6nFe4RqiPsnJusDETz&export=download&authuser=0&confirm=t&uuid=d5cbe99a-7752-429a-acc6-523486206e5b&at=AN8xHoqbqa-s94wnMJSCfgD8qxGM:1750259149349'


print(f"Attempting to load your uploaded file: {file_path}")

try:
    # Use pandas to read the CSV file from your Colab session.
    df = pd.read_csv(file_path)

    # --- Data Processing ---
    # The column names 'text' and 'label' are already correct, which is perfect.
    # We will select just these two columns to ensure the dataset is clean.
    df_final = df[['text', 'label']]

    # Drop any rows that might have missing text, just in case.
    df_final.dropna(inplace=True)

    # Convert the pandas DataFrame into a Hugging Face Dataset.
    dataset = Dataset.from_pandas(df_final)

    # Shuffle the dataset to ensure a good mix for training.
    dataset = dataset.shuffle(seed=42)

    print("\n✅ File loaded and processed successfully!")
    print(f"Total samples loaded: {len(dataset)}")
    print(dataset)
    #print("\nExample of Human-written text (label=0):")
    #print(dataset.filter(lambda example: example['label'] == 0)[0]['text'])
    #print("\nExample of AI-generated text (label=1):")
    #print(dataset.filter(lambda example: example['label'] == 1)[0]['text'])

except FileNotFoundError:
    print(f"\n❌ ERROR: FileNotFoundError.")
    print(f"Please make sure the file named '{file_path}' is uploaded to your Colab session.")
    print("You can check the file list by clicking the folder icon on the left sidebar.")
    print("Also, please double-check for any typos in the filename.")
except Exception as e:
    print(f"\nAn error occurred while loading or processing the file: {e}")


# Load RoBERTa Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Tokenize the Dataset
def tokenize_function(examples):
    # Truncate to 512 tokens, the max for roberta-base
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Remove the original text column as it's no longer needed after tokenization
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# This dataset has about 44868 examples. We'll use a larger subset for training.
train_size = 44000
eval_size = 868

small_train_dataset = tokenized_dataset.select(range(train_size))
small_eval_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))

print(f"Training dataset size: {len(small_train_dataset)}")
print(f"Evaluation dataset size: {len(small_eval_dataset)}")


# Configure and Run the Fine-Tuning
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

model_dir = "ai-human-detector-roberta"

training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

print("\nEvaluating the fine-tuned model...")
eval_results = trainer.evaluate()
print(f"Evaluation Accuracy: {eval_results['eval_accuracy']:.4f}")


# Save Model and Create Inference Pipeline
from transformers import pipeline

final_model_path = "./final_ai_human_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"Model saved to {final_model_path}")

# This pipeline will classify text as AI or HUMAN
detector = pipeline("text-classification", model=final_model_path, device=0)

print("\nInference pipeline created successfully!")


# Test the Fine-Tuned Detector
ai_text = "The synergistic application of blockchain technology and artificial intelligence is poised to redefine digital trust paradigms."
human_text = "Wow, this pizza is incredible! I haven't had one this good in ages."

def pretty_print_result(text, result):
    # LABEL_0 is Human, LABEL_1 is AI
    label_map = {'LABEL_0': 'HUMAN', 'LABEL_1': 'AI'}
    label = label_map[result[0]['label']]
    score = result[0]['score']
    print(f"Text: '{text[:80]}...'")
    print(f"--> Verdict: {label} (Confidence: {score:.4f})\n")

print("\n--- Classifying Test Cases ---")
pretty_print_result(ai_text, detector(ai_text))
pretty_print_result(human_text, detector(human_text))



#  Check input from User
check = input("Write your text sample here:")
pretty_print_result(check, detector(check))
