


# Part 2.1
import time
import torch
import datasets
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)  # default max_length=512

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'acc': acc, 'f1_score': f1}

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{trainable_params/total_params*100:.2f}% trainable')
    print(f'total params: {total_params}, trainable params: {trainable_params}')

data = datasets.load_dataset("tweet_eval", name="sentiment")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenized_data = data.map(tokenize_function, batched=True)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
# hyperparam args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir = './logs',
    logging_steps=100,
    load_best_model_at_end=True,
)
# Hugging Face's PyTorch Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['validation'],
    compute_metrics=compute_metrics,
)

# Initialize training
count_params(model)
start_time = time.time()
trainer.train()
end_time = time.time()
# Log info
training_time = end_time - start_time
print(f'Training spent: {training_time:.2f} seconds')
if torch.cuda.is_available():
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f'Max GPU Memory Allocated: {max_memory:.2f} GB')
else:
    print('GPU not used for training')
test_results = trainer.evaluate(tokenized_data['test'])
print(f'Test loss: {test_results["loss"]:.4f}')
print(f'Test acc: {test_results["eval_accuracy"]:.4f}')
print(f'Test F1 Score: {test_results["eval_f1_score"]:.4f}')







































