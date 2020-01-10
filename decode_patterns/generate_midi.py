import mido

from decode_patterns.data_conversion import parse_csv, build_track


def main():
    # примерно так мы csv превращаем в (почти) готовый dataset для использования
    # осталось его превратить в пары-картинки для pix2pix и будет замечательно
    patterns_file = "../patterns_pairs.tsv"
    dataset_pairs = parse_csv(patterns_file)

    # так можно сохранять мидишники в файл
    pattern_id = 232
    mid = build_track(dataset_pairs[pattern_id], tempo=dataset_pairs[pattern_id].tempo)
    mid.save(f"../midi/sample{pattern_id}.mid")

    # а так можно проигрывать его в наушниках
    port = mido.open_output(mido.get_output_names()[0])
    for msg in mid.play():
        port.send(msg)


if __name__ == '__main__':
    main()
