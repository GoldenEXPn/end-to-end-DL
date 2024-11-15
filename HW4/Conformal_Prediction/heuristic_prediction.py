import numpy as np

# Data preparation
softmax_outputs = np.load('softmax_outputs.npy')
correct_classes = np.load('correct_classes.npy')

# Data Split
split = 2000
calibration_softmax = softmax_outputs[:split]
calibration_labels = correct_classes[:split]
validation_softmax = softmax_outputs[split:]
validation_labels = correct_classes[split:]

random_scores = np.random.rand(*validation_softmax.shape)

def construct_naive_pred_set(random_output, q_hat):
    sorted_indices = np.argsort(random_output)[::-1]
    cumulative_sum = 0
    pred_set = []
    for idx in sorted_indices:
        cumulative_sum += random_output[idx]
        pred_set.append(idx)
        if cumulative_sum >= q_hat:
            break
    return pred_set

def construct_adaptive_pred_set(random_output, q_hat):
    sorted_indices = np.argsort(random_output)[::-1]
    cumulative_sum = 0
    pred_set = []
    for idx in sorted_indices:
        cumulative_sum += random_output[idx]
        pred_set.append(idx)
        if cumulative_sum >= q_hat:
            break
    return pred_set

# Compute Random Scores
random_calibration_scores = []
for i in range(len(calibration_softmax)):
    random_output = np.random.rand(calibration_softmax.shape[1])
    sorted_probs = np.sort(random_output)[::-1]
    cumulative_sum = np.cumsum(sorted_probs)
    score = cumulative_sum[np.argmax(sorted_probs)]
    random_calibration_scores.append(score)

# Compute quantile
alpha = 0.1
q_hat_random = np.quantile(random_calibration_scores, 1 - alpha)
print(f"Quantile (q_hat) using random scores: {q_hat_random:.4f}")

# Evaluate the empirical coverage
correct_naive = 0
correct_adaptive = 0
for i in range(len(validation_softmax)):
    random_output = random_scores[i]

    naive_pred_set = construct_naive_pred_set(random_output, q_hat_random)
    if validation_labels[i] in naive_pred_set:
        correct_naive += 1

    adaptive_pred_set = construct_adaptive_pred_set(random_output, q_hat_random)
    if validation_labels[i] in adaptive_pred_set:
        correct_adaptive += 1

empirical_coverage_naive_random = correct_naive / len(validation_softmax)
empirical_coverage_adaptive_random = correct_adaptive / len(validation_softmax)
print(f"Empirical Coverage (Naive with Random Scores): {empirical_coverage_naive_random:.4f}")
print(f'Prediction Set Size (Naive with Random Scores): {len(naive_pred_set)}')
print(f"Empirical Coverage (Adaptive with Random Scores): {empirical_coverage_adaptive_random:.4f}")
print(f'Prediction Set Size (Adaptive with Random Scores): {len(adaptive_pred_set)}')
