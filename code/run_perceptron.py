from perceptron import PerceptronClassifier
from data_loader import load_data
import random
import time
import matplotlib.pyplot as plt
import pickle
import os
import statistics

DATA_TYPE = input("Enter dataset to run on ('digit' or 'face'): ").strip().lower()
if DATA_TYPE not in ['digit', 'face']:
    raise ValueError("Invalid input. Please enter 'digit' or 'face'.")


MAX_ITERATIONS = 5
RUNS_PER_PERCENTAGE = 5


if DATA_TYPE == "digit":
    width, height = 28, 28
    train_img_path = '../data/digitdata/trainingimages'
    train_lbl_path = '../data/digitdata/traininglabels'
    test_img_path = '../data/digitdata/testimages'
    test_lbl_path = '../data/digitdata/testlabels'
    legal_labels = list(range(10))
elif DATA_TYPE == "face":
    width, height = 60, 70
    train_img_path = '../data/facedata/facedatatrain'
    train_lbl_path = '../data/facedata/facedatatrainlabels'
    test_img_path = '../data/facedata/facedatatest'
    test_lbl_path = '../data/facedata/facedatatestlabels'
    legal_labels = [0, 1]


with open(train_lbl_path, 'r') as f:
    num_train = len(f.readlines())
with open(test_lbl_path, 'r') as f:
    num_test = len(f.readlines())


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
    std_dev = statistics.stdev([1 - acc for acc in scores])

    avg_accuracies.append(avg_acc)
    train_times.append(duration)

    print(f"  Avg Accuracy: {avg_acc:.4f}, Time: {duration:.2f} sec")
    print(f"Standard Deviation of Error: {std_dev:.4f}")


    if p == 100:
        os.makedirs("../models", exist_ok=True)
        with open(f"../models/perceptron_{DATA_TYPE}.pkl", "wb") as f:
            pickle.dump(classifier.weights, f)
        print(f"Saved trained model weights to ../models/perceptron_{DATA_TYPE}.pkl")


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(percentages, avg_accuracies, marker='o')
plt.title(f"{DATA_TYPE.capitalize()} Accuracy vs Training Size")
plt.xlabel("Training Size (%)")
plt.ylabel("Average Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(percentages, train_times, marker='o', color='orange')
plt.title(f"{DATA_TYPE.capitalize()} Training Time vs Training Size")
plt.xlabel("Training Size (%)")
plt.ylabel("Time (s)")
plt.grid(True)

plt.tight_layout()
os.makedirs("../results", exist_ok=True)
plt.savefig(f'../results/{DATA_TYPE}.png')
plt.show()
