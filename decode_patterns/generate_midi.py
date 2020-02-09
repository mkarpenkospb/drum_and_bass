import mido
import numpy as np

from data_conversion import parse_csv, build_track
from data_conversion import DrumMelodyPair, Converter

# example of midi generation
def example():
    # примерно так мы csv превращаем в (почти) готовый dataset для использования
    # осталось его превратить в пары-картинки для pix2pix и будет замечательно
    patterns_file = "patterns.pairs.tsv"
    dataset_pairs = parse_csv(patterns_file)

    # так можно сохранять мидишники в файл
    pattern_id = 232
    mid = build_track(dataset_pairs[pattern_id], tempo=dataset_pairs[pattern_id].tempo)
    mid.save(f"../midi/sample{pattern_id}.mid")

    # а так можно проигрывать его в наушниках
    port = mido.open_output(mido.get_output_names()[0])
    for msg in mid.play():
        port.send(msg)

def example2():
    # примерно так мы csv превращаем в (почти) готовый dataset для использования
    # осталось его превратить в пары-картинки для pix2pix и будет замечательно
    patterns_file = "patterns.pairs.tsv"
    dataset_pairs = parse_csv(patterns_file)

    # так можно сохранять мидишники в файл
    pattern_id = 232
    pair = dataset_pairs[pattern_id]
    converter = Converter((32, 50))
    img = converter.convert_pair_to_numpy_image(pair)
    pair = converter.convert_numpy_image_to_pair(img)
    mid = build_track(pair, tempo=pair.tempo)
    mid.save(f"../midi/sample{pattern_id}.mid")

    # а так можно проигрывать его в наушниках
    port = mido.open_output(mido.get_output_names()[0])
    for msg in mid.play():
        port.send(msg)

def generate_midi(N = 1000):

    converter = Converter((64, 50))
    for i in range(N):
        drum = np.load(f"../midi/npy/drum{i + 1}.npy")
        bass = np.load(f"../midi/npy/bass{i + 1}.npy")
        img_dnb = np.concatenate((drum, bass), axis=1)
        print(f"drum: {drum.shape}, bass: {bass.shape}, img_dnb:{img_dnb.shape}")
        pair = converter.convert_numpy_image_to_pair(img_dnb)
        # pair = DrumMelodyPair(pair.drum_pattern, pair.melody, pair.tempo, pair.instrument, 2)
        mid = build_track(pair, tempo=pair.tempo)
        mid.save(f"../midi/npy/sample{i+1}.mid")

def main():
    generate_midi()
    # example2()

if __name__ == '__main__':
    main()
