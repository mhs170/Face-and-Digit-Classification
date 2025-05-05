import numpy as np
from types import SimpleNamespace

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

class Counter(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

def create_dummy_data():
    width, height = 2, 1  # simple 2-feature input
    data = []
    labels = []
    for x, y, label in [(0, 0, 0), (1, 0, 1), (0, 1, 0), (1, 1, 1)]:
        datum = Counter()
        datum[(0, 0)] = x
        datum[(1, 0)] = y
        data.append(counter_to_numpy(datum, width, height))
        labels.append(label_to_one_hot(label, 2))
    return data, labels, width, height

def train_dummy_nn():
    train_data, train_labels, width, height = create_dummy_data()
    input_size = width * height
    hidden_size_1 = 5
    hidden_size_2 = 4
    output_size = 2
    learn_rate = 0.1
    epochs = 1000

    w_i_h1 = np.random.uniform(-0.5, 0.5, (hidden_size_1, input_size))
    w_h1_h2 = np.random.uniform(-0.5, 0.5, (hidden_size_2, hidden_size_1))
    w_h2_o = np.random.uniform(-0.5, 0.5, (output_size, hidden_size_2))

    b_i_h1 = np.zeros((hidden_size_1, 1))
    b_h1_h2 = np.zeros((hidden_size_2, 1))
    b_h2_o = np.zeros((output_size, 1))

    for epoch in range(epochs):
        for img, label_vec in zip(train_data, train_labels):
            # Forward pass
            h1_pre = b_i_h1 + w_i_h1 @ img
            h1 = 1 / (1 + np.exp(-h1_pre))
            h2_pre = b_h1_h2 + w_h1_h2 @ h1
            h2 = 1 / (1 + np.exp(-h2_pre))
            o_pre = b_h2_o + w_h2_o @ h2
            o = 1 / (1 + np.exp(-o_pre))

            # Backward pass
            delta_o = o - label_vec
            w_h2_o -= learn_rate * delta_o @ h2.T
            b_h2_o -= learn_rate * delta_o

            delta_h2 = (w_h2_o.T @ delta_o) * (h2 * (1 - h2))
            w_h1_h2 -= learn_rate * delta_h2 @ h1.T
            b_h1_h2 -= learn_rate * delta_h2

            delta_h1 = (w_h1_h2.T @ delta_h2) * (h1 * (1 - h1))
            w_i_h1 -= learn_rate * delta_h1 @ img.T
            b_i_h1 -= learn_rate * delta_h1

    model = SimpleNamespace(w_i_h1=w_i_h1, w_h1_h2=w_h1_h2, w_h2_o=w_h2_o,
                            b_i_h1=b_i_h1, b_h1_h2=b_h1_h2, b_h2_o=b_h2_o)

    correct = 0
    for i, (img, label_vec) in enumerate(zip(train_data, train_labels)):
        h1 = 1 / (1 + np.exp(-(model.b_i_h1 + model.w_i_h1 @ img)))
        h2 = 1 / (1 + np.exp(-(model.b_h1_h2 + model.w_h1_h2 @ h1)))
        o = 1 / (1 + np.exp(-(model.b_h2_o + model.w_h2_o @ h2)))
        pred = np.argmax(o)
        true = np.argmax(label_vec)
        print(f"Input {i}: Prediction={pred}, True={true}")
        correct += pred == true

    print(f"\nTest Accuracy: {correct}/{len(train_data)} ({round(correct / len(train_data) * 100, 2)}%)")

if __name__ == "__main__":
    train_dummy_nn()
