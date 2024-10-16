

import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

dataset = load_dataset('tweet_eval', name='sentiment')
train_labels = dataset['train']['label']

majority = Counter(train_labels).most_common(1)[0][0]
test_labels = dataset['test']['label']
naive_preds = np.full_like(test_labels, majority)

accuracy = accuracy_score(test_labels, naive_preds)
f1 = f1_score(test_labels, naive_preds, average='weighted')
print(f'Naive Predictor with accuracy: {accuracy} and f1_score: {f1}')
