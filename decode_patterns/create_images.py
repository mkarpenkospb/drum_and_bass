import numpy as np
import torch

from decode_patterns.data_conversion import parse_csv, Converter, make_numpy_dataset

def create_images(file_name = "../patterns_pairs.tsv", img_size = (128, 50), limit=1000):
    return make_numpy_dataset(file_name = file_name, img_size=img_size, limit=limit)

def crop_data(drumnbass, drum):
    bass_only = []
    drum_only = []
    for x in drum:
        drum_only.append([i[:14] for i in x])
    for y in drumnbass:
        bass_only.append([i[14:] for i in y])
    return np.array(drum_only), np.array(bass_only)


def train_test(drum, bass, batch_size):
    limit = len(drum)
    train_size = int(0.8 * limit)
    test_size = limit - train_size
    indexes = np.concatenate((np.full(train_size, True),
                              np.full(test_size, False)), axis=None)
    np.random.shuffle(indexes)
    test_set = (torch.tensor(drum[np.invert(indexes)]), torch.tensor(bass[np.invert(indexes)]))
    train_set = (torch.tensor(drum[indexes]),
                 torch.tensor(bass[indexes]))
    train_set = (train_set[0].reshape([-1, batch_size, 128, 14]), train_set[1].reshape([-1, batch_size, 128, 36]))
    return train_set, test_set


if __name__ == "__main__":
    limit = 10
    drumnbass, drum = create_images(limit=limit)
    drum, bass = crop_data(drumnbass, drum)
    train_set, test_set = train_test(drum, bass, 2)
    a = [[i, 2 * i] for i in range(11)]
    b = [[i, 0.5 * i] for i in range(11)]
    train_set0 = (a, b)
    # for i, (X, Y) in enumerate(zip(*train_set)):
    #     print(f"i = {i}, X: {X}, Y: {Y}")
    print(f"train_set shape: {train_set[0].shape} and {train_set[1].shape}")
    print(f"test_set shape: {test_set[0].shape} and {test_set[1].shape}")