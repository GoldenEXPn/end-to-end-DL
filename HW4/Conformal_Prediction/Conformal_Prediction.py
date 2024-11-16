import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from PIL import Image


### Conformal Prediction Learning
'''
# Part 1 - Naive Algorithm
softmax_outputs = np.load("Conformal_Prediction/softmax_outputs.npy")
correct_classes = np.load("Conformal_Prediction/correct_classes.npy")

# Step 2: Split into calibration and validation sets
n_calibration = 2000
calibration_outputs = softmax_outputs[:n_calibration]
validation_outputs = softmax_outputs[n_calibration:]
calibration_labels = correct_classes[:n_calibration]
validation_labels = correct_classes[n_calibration:]

# Step 3: Define score function and calculate scores on the calibration set
def score_function(softmax_output, true_class):
    return softmax_output[true_class]

calibration_scores = np.array([score_function(out, label) for out, label in zip(calibration_outputs, calibration_labels)])

# Step 4: Compute the 1-quantile threshold
alpha = 0.1
q_hat = np.quantile(calibration_scores, 1 - alpha)

# Step 5: Create prediction sets on the validation data
def prediction_set(softmax_output, threshold):
    # Include classes with probability >= threshold
    pred_set = np.where(softmax_output >= 1 - threshold)[0]
    return pred_set

# Create prediction sets on the validation data using the modified function
prediction_sets = [prediction_set(out, q_hat) for out in validation_outputs]

# Calculate the average size of prediction sets
average_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
print(f"Average prediction set size: {average_set_size:.2f}")

# Step 6: Measure empirical coverage
correct_predictions = [label in pred_set for label, pred_set in zip(validation_labels, prediction_sets)]
coverage = np.mean(correct_predictions)
print(f"Empirical coverage on validation set: {coverage:.2f}")


# Step 7: Visualize example predictions
# Load the files from the example dataset
model_outputs = np.load("Conformal_Prediction/example_images/model_outputs.npy")
gt_classes = np.load("Conformal_Prediction/example_images/gt_classes.npy")
idx2cls = np.load("Conformal_Prediction/example_images/idx2cls.npy", allow_pickle=True)

# Create a dictionary to map class names to indices
class_name_to_index = {name: i for i, name in enumerate(idx2cls)}

random_indices = np.random.choice(len(model_outputs), 5, replace=False)

for idx in [3,18,21,4,12]:
    softmax_output = model_outputs[idx]
    true_label_name = gt_classes[idx]
    true_label = class_name_to_index[true_label_name]  # Convert class name to index
    pred_set = prediction_set(softmax_output, q_hat)
    
    # Load and display the image
    image_path = f"Conformal_Prediction/example_images/B{idx}.png"
    img = Image.open(image_path)
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True label: {true_label_name}")
    plt.show()
    
    # Display prediction set details
    pred_labels = [idx2cls[i] for i in pred_set]  # Direct indexing of class names
    pred_probs = softmax_output[pred_set]
    
    print(f"Example {idx + 1}:")
    print(f"  True label: {true_label_name}")
    print(f"  Prediction set: {pred_labels}")
    print(f"  Prediction probabilities: {pred_probs}")
    
    # Plot the softmax probabilities for all classes
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(softmax_output)), softmax_output)
    plt.xlabel("Class Index")
    plt.ylabel("Probability")
    plt.title(f"True label: {true_label_name} | Prediction set: {pred_labels}")
    plt.show()
'''
'''
# Part 2 - daptive Prediction Sets Algorithm
softmax_outputs = np.load("Conformal_Prediction/softmax_outputs.npy")
correct_classes = np.load("Conformal_Prediction/correct_classes.npy")

# Step 2: Split into calibration and validation sets
n_calibration = 2000
calibration_outputs = softmax_outputs[:n_calibration]
validation_outputs = softmax_outputs[n_calibration:]
calibration_labels = correct_classes[:n_calibration]
validation_labels = correct_classes[n_calibration:]

# Step 3: Define score function Adaptively
def adaptive_score_function(softmax_output, true_class):
    # Sort softmax probabilities in descending order
    sorted_indices = np.argsort(-softmax_output)
    sorted_probs = softmax_output[sorted_indices]
    cumulative_sum = 0.0
    for j, prob in enumerate(sorted_probs):
        cumulative_sum += prob
        if sorted_indices[j] == true_class:
            return cumulative_sum  # Return cumulative sum up to the true class

calibration_scores = np.array([adaptive_score_function(out, label) for out, label in zip(calibration_outputs, calibration_labels)])

# Step 4: Compute the 1-quantile threshold
alpha = 0.1
q_hat = np.quantile(calibration_scores, 1 - alpha)

# Step 5: Create prediction sets on the validation data Adaptively
def adaptive_prediction_set(softmax_output, threshold):
    # Sort probabilities in descending order
    sorted_indices = np.argsort(-softmax_output)
    sorted_probs = softmax_output[sorted_indices]
    
    cumulative_sum = 0.0
    pred_set = []
    for j, prob in enumerate(sorted_probs):
        cumulative_sum += prob
        pred_set.append(sorted_indices[j])
        if cumulative_sum >= threshold:
            break  # Stop once the cumulative sum reaches or exceeds the threshold
    
    return pred_set

# Create prediction sets on the validation data using the modified function
prediction_sets = [adaptive_prediction_set(out, q_hat) for out in validation_outputs]

# Calculate the average size of prediction sets
average_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
print(f"Average prediction set size: {average_set_size:.2f}")

# Step 6: Measure empirical coverage
correct_predictions = [label in pred_set for label, pred_set in zip(validation_labels, prediction_sets)]
coverage = np.mean(correct_predictions)
print(f"Empirical coverage on validation set: {coverage:.2f}")

# Step 7: Visualize example predictions
# Load the files from the example dataset
model_outputs = np.load("Conformal_Prediction/example_images/model_outputs.npy")
gt_classes = np.load("Conformal_Prediction/example_images/gt_classes.npy")
idx2cls = np.load("Conformal_Prediction/example_images/idx2cls.npy", allow_pickle=True)

# Create a dictionary to map class names to indices
class_name_to_index = {name: i for i, name in enumerate(idx2cls)}

# Repeat with the same examples tested in Part1
for idx in [3,18,21,4,12]:
    softmax_output = model_outputs[idx]
    true_label_name = gt_classes[idx]
    true_label = class_name_to_index[true_label_name]  # Convert class name to index
    pred_set = adaptive_prediction_set(softmax_output, q_hat)
    
    # Load and display the image
    image_path = f"Conformal_Prediction/example_images/B{idx}.png"
    img = Image.open(image_path)
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True label: {true_label_name}")
    plt.show()
    
    # Display prediction set details
    pred_labels = [idx2cls[i] for i in pred_set]  # Direct indexing of class names
    pred_probs = softmax_output[pred_set]
    
    print(f"Example {idx + 1}:")
    print(f"  True label: {true_label_name}")
    print(f"  Prediction set: {pred_labels}")
    print(f"  Prediction probabilities: {pred_probs}")
    
    # Plot the softmax probabilities for all classes
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(softmax_output)), softmax_output)
    plt.xlabel("Class Index")
    plt.ylabel("Probability")
    plt.title(f"True label: {true_label_name} | Prediction set: {pred_labels}")
    plt.show()
'''
# Part 4 
# Subpart 1

softmax_outputs = np.load("Conformal_Prediction/softmax_outputs.npy")
correct_classes = np.load("Conformal_Prediction/correct_classes.npy")

# Split into calibration and validation sets
n_calibration = 2000
calibration_outputs = softmax_outputs[:n_calibration]
validation_outputs = softmax_outputs[n_calibration:]
calibration_labels = correct_classes[:n_calibration]
validation_labels = correct_classes[n_calibration:]

# Define score function and calculate scores on the calibration set (using model's softmax scores)
def score_function(softmax_output, true_class):
    return softmax_output[true_class]

calibration_scores = np.array([score_function(out, label) for out, label in zip(calibration_outputs, calibration_labels)])

# Compute the 1-quantile threshold for model-based scores
alpha = 0.1
q_hat_model = np.quantile(calibration_scores, 1 - alpha)

# Generate prediction sets with model softmax scores on validation data
def prediction_set(output, threshold):
    pred_set = np.where(output >= 1 - threshold)[0]
    return pred_set

# Prediction sets and evaluation using model-based softmax scores
prediction_sets_model = [prediction_set(out, q_hat_model) for out in validation_outputs]
average_set_size_model = np.mean([len(pred_set) for pred_set in prediction_sets_model])
coverage_model = np.mean([label in pred_set for label, pred_set in zip(validation_labels, prediction_sets_model)])
print(f"Model-based softmax scores -> Average prediction set size: {average_set_size_model:.2f}, Coverage: {coverage_model:.2f}")

# Generate random scores for calibration and validation data
random_calibration_scores = np.random.rand(*calibration_outputs.shape)
random_calibration_scores /= random_calibration_scores.sum(axis=1, keepdims=True)

random_scores = np.random.rand(*validation_outputs.shape)
random_scores /= random_scores.sum(axis=1, keepdims=True)  # Normalize to sum to 1

# Calculate a new threshold (q_hat_random) based on random calibration scores
random_scores_calibration = np.array([np.max(out) for out in random_calibration_scores])
q_hat_random = np.quantile(random_scores_calibration, 1 - alpha)

# Create prediction sets on the validation data with random scores
prediction_sets_random = [prediction_set(out, q_hat_random) for out in random_scores]

# Calculate the average size of prediction sets with random scores
average_set_size_random = np.mean([len(pred_set) for pred_set in prediction_sets_random])
print(f"Random scores -> Average prediction set size: {average_set_size_random:.2f}")

# Measure empirical coverage with random scores
correct_predictions_random = [label in pred_set for label, pred_set in zip(validation_labels, prediction_sets_random)]
coverage_random = np.mean(correct_predictions_random)
print(f"Random scores -> Empirical coverage on validation set: {coverage_random:.2f}")

# Step 7: Visualize example predictions with random scores
# Load example dataset for visualization
model_outputs = np.load("Conformal_Prediction/example_images/model_outputs.npy")
gt_classes = np.load("Conformal_Prediction/example_images/gt_classes.npy")
idx2cls = np.load("Conformal_Prediction/example_images/idx2cls.npy", allow_pickle=True)

# Create a dictionary to map class names to indices
class_name_to_index = {name: i for i, name in enumerate(idx2cls)}

# Choose specific indices for visualization
for idx in [3, 18, 21, 4, 12]:
    # Generate new random scores for visualization
    random_output = np.random.rand(len(model_outputs[idx]))
    random_output /= random_output.sum()  # Normalize to sum to 1
    
    true_label_name = gt_classes[idx]
    true_label = class_name_to_index[true_label_name]  # Convert class name to index
    pred_set_random = prediction_set(random_output, q_hat_random)
    
    # Load and display the image
    image_path = f"Conformal_Prediction/example_images/B{idx}.png"
    img = Image.open(image_path)
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True label: {true_label_name}")
    plt.show()
    
    # Display prediction set details
    pred_labels_random = [idx2cls[i] for i in pred_set_random]  # Direct indexing of class names
    pred_probs_random = random_output[pred_set_random]
    
    print(f"Example {idx + 1}:")
    print(f"  True label: {true_label_name}")
    print(f"  Prediction set with random scores: {pred_labels_random}")
    print(f"  Prediction probabilities with random scores: {pred_probs_random}")
    
    # Plot the random probabilities for all classes
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(random_output)), random_output)
    plt.xlabel("Class Index")
    plt.ylabel("Probability")
    plt.title(f"True label: {true_label_name} | Prediction set with random scores: {pred_labels_random}")
    plt.show()
