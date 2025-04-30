
from perceptron import PerceptronClassifier
from data_loader import load_data
import random
import time
import matplotlib.pyplot as plt

DATA_TYPE = "face"  # Change to "face" for face data, change to "digit" for digit data
MAX_ITERATIONS = 5
RUNS_PER_PERCENTAGE = 5


if DATA_TYPE == "digit":
    width, height = 28, 28
    train_img_path = '../data/digitdata/trainingimages'
    train_lbl_path = '../data/digitdata/traininglabels'
    test_img_path = '../data/digitdata/testimages'
    test_lbl_path = '../data/digitdata/testlabels'
    legal_labels = list(range(10))
    num_train = 5000
    num_test = 1000
elif DATA_TYPE == "face":
    width, height = 60, 70
    train_img_path = '../data/facedata/facedatatrain'
    train_lbl_path = '../data/facedata/facedatatrainlabels'
    test_img_path = '../data/facedata/facedatatest'
    test_lbl_path = '../data/facedata/facedatatestlabels'
    legal_labels = [0, 1]  # Face or not face
    num_train = 451
    num_test = 150
else:
    raise ValueError("Unsupported DATA_TYPE. Use 'digit' or 'face'.")


print("Loading data...")
train_data, train_labels = load_data(train_img_path, train_lbl_path, num_train, width, height)
test_data, test_labels = load_data(test_img_path, test_lbl_path, num_test, width, height)
print("Done.")


percentages = list(range(10, 101, 10))
avg_accuracies = []
train_times = []

for p in percentages:
    print(f"Training on {p}% of data...")
    size = int(len(train_data) * (p / 100))
    scores = []
    start_time = time.time()

    for _ in range(RUNS_PER_PERCENTAGE):
        
        subset = random.sample(list(zip(train_data, train_labels)), size)
        subset_data, subset_labels = zip(*subset)

        
        classifier = PerceptronClassifier(legal_labels, MAX_ITERATIONS)
        classifier.train(subset_data, subset_labels, [], [])

        
        predictions = classifier.classify(test_data)
        acc = sum([pred == true for pred, true in zip(predictions, test_labels)]) / len(test_labels)
        scores.append(acc)

    avg_acc = sum(scores) / len(scores)
    duration = time.time() - start_time

    avg_accuracies.append(avg_acc)
    train_times.append(duration)

    print(f"  Avg Accuracy: {avg_acc:.4f}, Time: {duration:.2f} sec")


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(percentages, avg_accuracies, marker='o')
plt.title("Accuracy vs Training Data Size")
plt.xlabel("Training Size (%)")
plt.ylabel("Average Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(percentages, train_times, marker='o', color='orange')
plt.title("Training Time vs Training Data Size")
plt.xlabel("Training Size (%)")
plt.ylabel("Time (s)")
plt.grid(True)

plt.tight_layout()
plt.show()