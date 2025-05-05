from data_loader import load_data
import numpy as np
import matplotlib.pyplot as plt
import time
from util import FixedRandom

MAX_ITERATIONS = 5
RUNS_PER_PERCENTAGE = 5
percentages = range(10, 101, 10)
RUNS_PER_PERCENTAGE = 5

def counter_to_numpy(counter, width, height):
    arr = np.zeros((width * height, 1))
    for (x, y), value in counter.items():
        index = y * width + x   
        arr[index][0] = value
    return arr

def label_to_one_hot(label, num_classes):
    vec = np.zeros((num_classes, 1))
    vec[label][0] = 1
    return vec

def evaluate_accuracy(model, data, labels):
    w_i_h1, w_h1_h2, w_h2_o, b_i_h1, b_h1_h2, b_h2_o = model
    correct = 0
    for img, label_vec in zip(data, labels):
        h1 = 1 / (1 + np.exp(-(b_i_h1 + w_i_h1 @ img)))
        h2 = 1 / (1 + np.exp(-(b_h1_h2 + w_h1_h2 @ h1)))
        o = 1 / (1 + np.exp(-(b_h2_o + w_h2_o @ h2)))
        if np.argmax(o) == np.argmax(label_vec):
            correct += 1
    return round((correct / len(data)) * 100, 2)

def main():
    data_type = input("Enter data type (digit/face): ").strip().lower()
    if data_type == "digit":
        width, height = 28, 28
        train_img_path = '../data/digitdata/trainingimages'
        train_lbl_path = '../data/digitdata/traininglabels'
        test_img_path = '../data/digitdata/testimages'
        test_lbl_path = '../data/digitdata/testlabels'
        legal_labels = list(range(10))
        num_train = 5000
        num_test = 1000
        epochs = 5
        learn_rate = 0.01
        hidden_size = 20
    elif data_type == "face":
        width, height = 60, 70
        train_img_path = '../data/facedata/facedatatrain'
        train_lbl_path = '../data/facedata/facedatatrainlabels'
        test_img_path = '../data/facedata/facedatatest'
        test_lbl_path = '../data/facedata/facedatatestlabels'
        legal_labels = [0, 1]
        num_train = 451
        num_test = 150
        hidden_size = 20
        epochs = 30
        learn_rate = 0.02
    else:
        raise ValueError("Unsupported DATA_TYPE.")


    print("Loading data...")
    train_data, train_labels = load_data(train_img_path, train_lbl_path, num_train, width, height)
    test_data, test_labels = load_data(test_img_path, test_lbl_path, num_test, width, height)
    print("Done.")


    input_size = width * height
    output_size = len(legal_labels)
    train_data = [counter_to_numpy(img, width, height) for img in train_data]
    train_labels = [label_to_one_hot(l, output_size) for l in train_labels]
    test_data = [counter_to_numpy(img, width, height) for img in test_data]
    test_labels = [label_to_one_hot(l, output_size) for l in test_labels]


    avg_accuracies = []
    std_accuracies = []
    avg_times = []
    rng = FixedRandom().random

    for percent in percentages:
        print(f"\nTraining on {percent}% of data...")
        N = int(len(train_data) * percent / 100)

        scores = []
        start = time.time()
        for _ in range(RUNS_PER_PERCENTAGE):
            indices = rng.sample(range(len(train_data)), N)
            subset_data = [train_data[i] for i in indices]
            subset_labels = [train_labels[i] for i in indices]

            hidden_size_1 = hidden_size
            hidden_size_2 = hidden_size

            w_i_h1 = np.random.uniform(-0.5, 0.5, (hidden_size_1, input_size))
            b_i_h1 = np.zeros((hidden_size_1, 1))

            w_h1_h2 = np.random.uniform(-0.5, 0.5, (hidden_size_2, hidden_size_1))
            b_h1_h2 = np.zeros((hidden_size_2, 1))

            w_h2_o = np.random.uniform(-0.5, 0.5, (output_size, hidden_size_2))
            b_h2_o = np.zeros((output_size, 1))
            
        
            for epoch in range(epochs):
                for img, label_vec in zip(subset_data, subset_labels):
                    #Forward pass
                    h1_pre = b_i_h1 + w_i_h1 @ img
                    h1 = 1 / (1 + np.exp(-h1_pre))

                    h2_pre = b_h1_h2 + w_h1_h2 @ h1
                    h2 = 1 / (1 + np.exp(-h2_pre))

                    o_pre = b_h2_o + w_h2_o @ h2
                    o = 1 / (1 + np.exp(-o_pre))

                    #Back propagation
                    delta_o = o - label_vec
                    w_h2_o += -learn_rate * delta_o @ h2.T
                    b_h2_o += -learn_rate * delta_o


                    delta_h2 = (w_h2_o.T @ delta_o) * (h2 * (1 - h2))
                    w_h1_h2 += -learn_rate * delta_h2 @ h1.T
                    b_h1_h2 += -learn_rate * delta_h2


                    delta_h1 = (w_h1_h2.T @ delta_h2) * (h1 * (1 - h1))
                    w_i_h1 += -learn_rate * delta_h1 @ img.T
                    b_i_h1 += -learn_rate * delta_h1
            acc = evaluate_accuracy((w_i_h1, w_h1_h2, w_h2_o, b_i_h1, b_h1_h2, b_h2_o), test_data, test_labels)
            scores.append(acc)        
        end = time.time()
        
        duration = round(end - start, 2) 
        avg_acc = round(sum(scores) / len(scores), 2)   
        std_acc = round(np.std(scores), 2)
        print(f"  Avg Accuracy: {avg_acc:.2f}%, Std Dev: {std_acc:.2f}%, Time: {duration:.2f} sec")
        avg_accuracies.append(avg_acc)
        avg_times.append(duration)
    print(f"Total time for all runs: {sum(avg_times):.2f} sec")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(percentages, avg_accuracies, marker='o')
    plt.title("Average Test Accuracy vs % Training Data")
    plt.xlabel("% Training Data Used")
    plt.ylabel("Average Accuracy (%)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(percentages, avg_times, marker='o', color='orange')
    plt.title("Total Training Time vs % Training Data")
    plt.xlabel("% Training Data Used")
    plt.ylabel("Total Time (sec)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    while True:
        index = int(input(f"Enter a number (0-{len(train_data)-1}): "))
        if index == -1:
            break
        img = train_data[index]
        raw_img = img.reshape((height, width))  # For display


        h1_pre = b_i_h1 + w_i_h1 @ img
        h1 = 1 / (1 + np.exp(-h1_pre))
        h2_pre = b_h1_h2 + w_h1_h2 @ h1
        h2 = 1 / (1 + np.exp(-h2_pre))
        o_pre = b_h2_o + w_h2_o @ h2
        o = 1 / (1 + np.exp(-o_pre))

        plt.imshow(raw_img, cmap="Greys")
        if data_type == "face":
            prediction = "Face" if o.argmax() == 1 else "Not Face"
            answer = "Face" if train_labels[index].argmax() == 1 else "Not Face"
        else:
            prediction = o.argmax()
            answer = train_labels[index].argmax()
        plt.title(f"Prediction: {prediction}, Answer: {answer}")
        plt.show()

if __name__ == "__main__":
    main()