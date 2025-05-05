import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import time
import matplotlib.pyplot as plt

def load_data(image_file, label_file, num_items, width, height):
    with open(image_file, 'r') as img_f, open(label_file, 'r') as lbl_f:
        image_lines = img_f.readlines()
        label_lines = lbl_f.readlines()

    images = []
    for i in range(0, len(image_lines), height):
        image = image_lines[i:i+height]
        flat_image = []
        for line in image:
            for char in line.strip('\n'):
                flat_image.append(0 if char == ' ' else 1)
        images.append(flat_image)

    labels = [int(lbl.strip()) for lbl in label_lines]
    X = torch.tensor(images[:num_items], dtype=torch.float32)
    y = torch.tensor(labels[:num_items], dtype=torch.long)
    X /= 1.0
    return X, y

class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

def reset_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.1, 0.1)
        nn.init.zeros_(m.bias)

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

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
        epochs = 10
        learn_rate = 0.02
    else:
        raise ValueError("Unsupported data type.")

    print("Loading data...")
    X_train, y_train = load_data(train_img_path, train_lbl_path, num_train, width, height)
    X_test, y_test = load_data(test_img_path, test_lbl_path, num_test, width, height)
    print("Done.")

    input_size = width * height
    output_size = len(legal_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    percentages = list(range(10, 101, 10))
    avg_accuracies = []
    std_accuracies = []
    avg_times = []

    for pct in percentages:
        print(f"Training on {pct}% of data...")
        N = int(len(X_train) * pct / 100)
        accs = []
        start = time.time()
        for _ in range(5):
            indices = random.sample(range(len(X_train)), N)
            X_sub = X_train[indices]
            y_sub = y_train[indices]

            train_loader = DataLoader(TensorDataset(X_sub, y_sub), batch_size=64, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

            model = ThreeLayerNN(input_size, 128, 64, output_size).to(device)
            model.apply(reset_weights)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learn_rate)

            acc = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs)
            accs.append(acc * 100)
        end = time.time()
        duration = round(end - start, 2)
        avg_acc = round(np.mean(accs), 2)
        std_acc = round(np.std(accs), 2)  # Already in percentage since accs is multiplied above

        print(f"  Avg Accuracy: {avg_acc:.2f}%, Std Dev: {std_acc:.2f}%, Time: {duration:.2f} sec")
        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_times.append(duration)


    
    # Plot results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(percentages, avg_accuracies, marker='o')
    plt.title("Average Test Accuracy vs % Training Data (PyTorch)")
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
    plt.show()


    # Visualization loop
    while True:
        user_input = input(f"Enter a number (0–{len(X_train)-1}) to visualize an image, or type 'exit' to quit: ").strip()
        if user_input.lower() in ['exit', '-1']:
            print("Exiting visualization.")
            break
        if not user_input.isdigit():
            print("Invalid input. Please enter a valid number or 'exit'.")
            continue
        index = int(user_input)
        if not (0 <= index < len(X_train)):
            print(f"Number out of range. Please enter a number between 0 and {len(X_train)-1}.")
            continue

        img_tensor = X_train[index]
        label = y_train[index].item()

        with torch.no_grad():
            model.eval()
            img_input = img_tensor.view(1, -1).to(device)
            output = model(img_input)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        img = img_tensor.view(height, width).cpu().numpy()
        plt.imshow(img, cmap="Greys")

        if data_type == "face":
            prediction = "Face" if pred_class == 1 else "Not Face"
            answer = "Face" if label == 1 else "Not Face"
        else:
            prediction = pred_class
            answer = label

        plt.title(f"Prediction: {prediction}, Answer: {answer}")
        plt.show()

if __name__ == '__main__':
    main()
