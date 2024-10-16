

# Part 3
import time
import torch
import datasets
from sklearn.metrics import accuracy_score, f1_score
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding


# noinspection DuplicatedCode
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
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
data_collator = DataCollatorWithPadding(tokenizer)  # Dynamic padding
tokenized_data = data.map(tokenize_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=3)

# Inject lora
peft_config=LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8)
model = get_peft_model(model, peft_config)

# hyperparam args
# noinspection DuplicatedCode
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy='epoch',
    save_strategy='epoch',
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
    data_collator=data_collator,
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
print(f'Test loss: {test_results["eval_loss"]:.4f}')
print(f'Test acc: {test_results["eval_acc"]:.4f}')
print(f'Test F1 Score: {test_results["eval_f1_score"]:.4f}')







































