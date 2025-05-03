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
    w_i_h, w_h_o, b_i_h, b_h_o = model
    correct = 0
    for img, label_vec in zip(data, labels):
        h = 1 / (1 + np.exp(-(b_i_h + w_i_h @ img)))
        o = 1 / (1 + np.exp(-(b_h_o + w_h_o @ h)))
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
    hidden_size = 5
    output_size = 2
    learn_rate = 0.1
    epochs = 1000

    w_i_h = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
    w_h_o = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
    b_i_h = np.zeros((hidden_size, 1))
    b_h_o = np.zeros((output_size, 1))

    for epoch in range(epochs):
        for img, label_vec in zip(train_data, train_labels):
            h_pre = b_i_h + w_i_h @ img 
            h = 1 / (1 + np.exp(-h_pre))  
            o_pre = b_h_o + w_h_o @ h 
            o = 1 / (1 + np.exp(-o_pre))

            delta_o = o - label_vec
            w_h_o -= learn_rate * delta_o @ h.T
            b_h_o -= learn_rate * delta_o

            delta_h = (w_h_o.T @ delta_o) * (h * (1 - h))
            w_i_h -= learn_rate * delta_h @ img.T
            b_i_h -= learn_rate * delta_h

    model = SimpleNamespace(w_i_h=w_i_h, w_h_o=w_h_o, b_i_h=b_i_h, b_h_o=b_h_o)
    correct = 0
    for i, (img, label_vec) in enumerate(zip(train_data, train_labels)):
        h = 1 / (1 + np.exp(-(model.b_i_h + model.w_i_h @ img)))
        o = 1 / (1 + np.exp(-(model.b_h_o + model.w_h_o @ h)))
        pred = np.argmax(o)
        true = np.argmax(label_vec)
        print(f"Input {i}: Prediction={pred}, True={true}")
        correct += pred == true

    print(f"\nTest Accuracy: {correct}/{len(train_data)} ({round(correct / len(train_data) * 100, 2)}%)")

if __name__ == "__main__":
    train_dummy_nn()