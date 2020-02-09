import numpy as np
import torch

from decode_patterns.data_conversion import parse_csv, Converter, make_numpy_dataset

def create_images(file_name = "../patterns_pairs.tsv", img_size = (128, 50), limit=1000):
    return make_numpy_dataset(patterns_file = file_name, img_size=img_size, limit=limit)

def crop_data(drumnbass):
    bass_only = []
    drum_only = []
    for x in drumnbass:
        drum_only.append([i[:14] for i in x])
        bass_only.append([i[14:] for i in x])
    return np.array(drum_only), np.array(bass_only)


def train_test(drum, bass, batch_size, img_size=(128, 50)):
    limit = len(drum)
    train_size = int(0.8 * limit)
    test_size = limit - train_size
    indexes = np.concatenate((np.full(train_size, True),
                              np.full(test_size, False)), axis=None)
    np.random.shuffle(indexes)
    test_set = (torch.tensor(drum[np.invert(indexes)]),
                torch.tensor(bass[np.invert(indexes)]))
    train_set = (torch.tensor(drum[indexes]),
                 torch.tensor(bass[indexes]))
    train_set = (train_set[0].reshape([-1, batch_size, img_size[0], 14]),
                 train_set[1].reshape([-1, batch_size, img_size[0], 36]))
    return train_set, test_set


if __name__ == "__main__":
    limit = 10
    drumnbass, _ = create_images(limit=limit, img_size=(128, 50))
    drum, bass = crop_data(drumnbass)
    train_set, test_set = train_test(drum, bass, batch_size=2, img_size=(128, 50))
    # a = [[i, 2 * i] for i in range(11)]
    # b = [[i, 0.5 * i] for i in range(11)]
    # train_set0 = (a, b)
    for i, (X, Y) in enumerate(zip(*train_set)):
        print(f"i = {i}, X: {X}, Y: {Y}")
    print(f"train_set shape: {train_set[0].shape} and {train_set[1].shape}")
    print(f"test_set shape: {test_set[0].shape} and {test_set[1].shape}")