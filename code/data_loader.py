from util import Counter

def load_data(image_file, label_file, num_images, width, height):
    images = []
    labels = []

    with open(image_file, 'r') as img_f, open(label_file, 'r') as lbl_f:
        image_lines = img_f.readlines()
        label_lines = lbl_f.readlines()

        for i in range(num_images):
            datum = Counter()
            for y in range(height):
                line = image_lines[i * height + y]
                for x in range(width):
                    char = line[x]
                    key = (x, y)
                    if char == ' ':
                        datum[key] = 0
                    elif char in ('+', '#'):
                        datum[key] = 1
            images.append(datum)
            labels.append(int(label_lines[i].strip()))

    return images, labels
