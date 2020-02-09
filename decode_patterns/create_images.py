import numpy as np
import torch
from sklearn.utils import shuffle
from decode_patterns.data_conversion import parse_csv, Converter, make_numpy_dataset


def create_images(file_name = "../patterns_pairs.tsv", img_size = (128, 50), limit=1000):
    return make_numpy_dataset(file_name = file_name, img_size=img_size, limit=limit)


def crop_data(drumnbass):
    bass_only = []
    drum_only = []
    for x in drumnbass:
        drum_only.append([i[:14] for i in x])
        bass_only.append([i[14:] for i in x])
    return np.array(drum_only), np.array(bass_only)


def train_test(drum, bass, tempo, batch_size, img_size=(128, 50)):
    limit = len(drum)
    train_size = int(0.8 * limit)
    test_size = limit - train_size
    indexes = np.concatenate((np.full(train_size, True),
                              np.full(test_size, False)), axis=None)
    np.random.shuffle(indexes)
    test_set = ((torch.tensor(drum[np.invert(indexes)]),
                torch.tensor(bass[np.invert(indexes)])), tempo[np.invert(indexes)])
    d_train, b_train, t_train = drum[indexes], bass[indexes], tempo[indexes]
    d_train, b_train, t_train = shuffle(d_train, b_train, t_train, random_state=0)
    train_set = ((torch.tensor(d_train).reshape([-1, batch_size, 1, 14 * img_size[0] + 16]),
                 torch.tensor(b_train).reshape([-1, batch_size, 1, 36 * img_size[0]])), t_train)
    return train_set, test_set


if __name__ == "__main__":
    limit = 10
    # drumnbass, _ =
    drum, bass, tempo = create_images(limit=limit, img_size=(128, 50))
    (train_set, train_t), (test_set, test_t) = train_test(drum, bass, tempo, batch_size=2, img_size=(128, 50))
    train_set[0][0].reshape([-1, 1, 128 * 14 + 16])
    a = 5
    # # a = [[i, 2 * i] for i in range(11)]
    # # b = [[i, 0.5 * i] for i in range(11)]
    # # train_set0 = (a, b)
    # for i, (X, Y) in enumerate(zip(*train_set)):
    #     print(f"i = {i}, X: {X}, Y: {Y}")
    # print(f"train_set shape: {train_set[0].shape} and {train_set[1].shape}")
    # print(f"test_set shape: {test_set[0].shape} and {test_set[1].shape}")