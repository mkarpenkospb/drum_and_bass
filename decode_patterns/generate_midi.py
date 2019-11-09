import mido
import csv
import numpy as np
from mido import MidiFile, MidiTrack, Message
from collections import namedtuple

# это индексы барабанных инструменты, которые мы по умолчанию подразумеваем
# определёнными в датасете
ALLOWED_PITCH_LIST = list(reversed([36, 38, 42, 41, 45, 46, 48, 49, 51, 58, 60, 61, 62, 64]))
ALLOWED_PITCH_NP = np.array(ALLOWED_PITCH_LIST)
# дефолтная длительность четверти в библиотеке MIDO для midi-файлов
MIDO_DEFAULT_QUARTER_LENGTH = 480


# Наш датасет состоит из множества пар
# Парой является (drum_pattern, melody)
# Но нам интересна и некоторая дополнительная информация:
#  - tempo -- темп
#  - instrument -- инструмент, которым сыграли мелодию
#  - denominator -- делитель темпа в мелодии (например, если denominator=2, то
#    длительность при отсчёте долей в мелодии в два раза короче, чем в барабанной партии)
DrumMelodyPair = namedtuple("DrumMelodyPair", ["drum_pattern", "melody", "tempo", "instrument", "denominator"])


def parse_csv(tsv_file_path: str):
    drum_melody_pairs = []
    with open(tsv_file_path) as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
        for row in tsv_reader:
            pattern = [
                       np.array([bool(int(i)) for i in j]) for j in
                       [bin(int(x))[2:].rjust(14, '0') for x in row[10].split(',')]
                      ]
            # tempo.append(int(row[3]))
            # drum_inst = [int(x) for x in row[5].split(',')]
            tempo = int(row[3])
            drum_pattern = [list(ALLOWED_PITCH_NP[i]) for i in pattern]
            # добавим инструмент, который исполняет мелодию
            instrument = int(row[13])
            pairs = eval(row[15])
            # обработка мелодии
            denominator = int(row[8])
            melody = [[] for _ in range(32 * denominator)]
            for i in pairs:
                if i[0] >= 32 * denominator:
                    continue
                melody[i[0]].append(i[1])

            drum_melody_pairs.append(DrumMelodyPair(drum_pattern, melody, tempo, instrument, denominator))
    return drum_melody_pairs


# по паре мы можем создать проигрываемый midi-файл
def build_track(drum_bass_pair: DrumMelodyPair,
                repeats: int = 4, tempo: int = None):
    # для регулировки темпа используем коэффициент растяжения
    k = 1
    # растягивать будем дефолтную длительность шестнадцатой доли в темпе 120 bpm
    time_quant = MIDO_DEFAULT_QUARTER_LENGTH / 4

    # не самый лучший способ регулировки темпа
    # TODO сделать регулировку темпа через метасобытие
    def adjust_tempo():
        if not tempo:
            bpm = drum_bass_pair.tempo
        else:
            bpm = tempo
        bpm_default = 120
        global k, time_quant
        k = bpm_default / bpm
        time_quant = int((MIDO_DEFAULT_QUARTER_LENGTH / 4) * k)

    # регулируем темп
    adjust_tempo()

    # этап 1 -- записать в миди-файл барабанную партию
    midi_file = MidiFile(type=1)  # создаём midi-файл
    track = MidiTrack()  # создаём барабанный трек в midi-файле
    midi_file.tracks.append(track)

    time = 0

    while repeats > 0:
        repeats -= 1
        for i in drum_bass_pair.drum_pattern:
            if not i:
                time += time_quant
                continue

            for j in i:
                track.append(Message('note_on', note=j, velocity=127, time=int(time), channel=9))
                time = 0
            time = time_quant

    # этап 2 -- записать в миди-файл барабанную партию
    track2 = MidiTrack()
    midi_file.tracks.append(track2)
    # метаинформацией изменяем голос инструмента специальным midi-сообщением
    track2.append(Message('program_change', program=drum_bass_pair.instrument, time=0, channel=2))

    time = 0
    last_notes = []
    while repeats > 0:
        repeats -= 1
        for i in (drum_bass_pair.melody):
            if not i:
                # все ноты сокращаются в denominator раз
                time += time_quant / drum_bass_pair.denominator
                continue

            for j in last_notes:
                track2.append(Message('note_off', note=j, velocity=127, time=int(time), channel=2))
                time = 0
            for j in i:
                track2.append(Message('note_on', note=j, velocity=127, time=int(time), channel=2))
                time = 0
            # все ноты сокращаются в denominator раз
            time = time_quant / drum_bass_pair.denominator
            last_notes = i

    for j in last_notes:
        track2.append(Message('note_off', note=j, velocity=127, time=int(time), channel=2))
        time = 0

    return midi_file


def main():
    # примерно так мы csv превращаем в (почти) готовый dataset для использования
    # осталось его превратить в пары-картинки для pix2pix и будет замечательно
    patterns_file = "test.tsv"
    dataset_pairs = parse_csv(patterns_file)

    # так можно сохранять мидишники в файл
    id = 3
    mid = build_track(dataset_pairs[3])
    mid.save(f"../midi/sample{id}.mid")

    # а так можно проигрывать его в наушниках
    port = mido.open_output(mido.get_output_names()[0])
    for msg in mid.play():
        port.send(msg)

if __name__ == '__main__':
    main()
