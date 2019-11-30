import csv
import numpy as np
from mido import MidiFile, MidiTrack, Message
from collections import namedtuple
import random

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


def parse_csv(tsv_file_path: str, limit=1000):
    drum_melody_pairs = []
    with open(tsv_file_path) as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
        for row in tsv_reader:
            if limit <= 0:
                return drum_melody_pairs
            limit -= 1

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
        nonlocal k, time_quant
        k = bpm_default / bpm
        time_quant = int((MIDO_DEFAULT_QUARTER_LENGTH / 4) * k)

    # регулируем темп
    adjust_tempo()

    # этап 1 -- записать в миди-файл барабанную партию
    midi_file = MidiFile(type=1)  # создаём midi-файл
    track = MidiTrack()  # создаём барабанный трек в midi-файле
    midi_file.tracks.append(track)

    time = 0

    repeat = repeats
    while repeat > 0:
        repeat -= 1
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
    repeat = repeats
    last_notes = []
    while repeat > 0:
        repeat -= 1
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

# создаём класс для конвертации музыкальный пар в массивы numpy и наоборот
# В классе, соответственно, два метода. При создании класса мы указываем
# параметры конвертации: размер сетки-изображения и максимальное кол-во голосов
# в мелодии.
class Converter:
    def __init__(self, grid_size=(128, 50)):
        self.grid_size = grid_size
        self.time_count = grid_size[0]
        self.drum_range = len(ALLOWED_PITCH_LIST)
        self.instrument_range = grid_size[1] - self.drum_range
        if (self.instrument_range <= 0):
            raise ValueError("Cannot instantiate converter with zero-bandwidth melody channel")

    # TODO -- check + test implementation
    def convert_pair_to_numpy_image(self, drum_bass_pair: DrumMelodyPair) -> np.array:
        # range = self.grid_size
        pattern_track = np.zeros(self.grid_size, dtype=np.float32)
        # Step 1. Fill in drum part
        # precount indexes
        idx = {}
        for i in range(len(ALLOWED_PITCH_LIST)):
            idx[ALLOWED_PITCH_LIST[i]] = i
        #uniformly distribute drum_pattern on pattern track
        drum_length = len(drum_bass_pair.drum_pattern)
        for k in range(drum_length):
            drum_col = drum_bass_pair.drum_pattern[k]
            i = int(k * self.time_count / drum_length)
            for v in drum_col:
                j = idx[v]
                if j >= self.drum_range:
                    continue
                pattern_track[i, j] = 1

        # Step 2. Fill in melody part
        # find minimum pitch of the melody
        melody_length = len(drum_bass_pair.melody)
        min_melody_pitch = None
        for melody_col in drum_bass_pair.melody:
            # if not melody_col:
            #     if not min_melody_pitch:
            if melody_col:
                if not min_melody_pitch:
                    min_melody_pitch = min(melody_col)
                min_melody_pitch = min(min_melody_pitch, min(melody_col))
        # minimum found -- fill in array
        for k in range(melody_length):
            melody_col = drum_bass_pair.melody[k]
            i = int(k * self.time_count / melody_length)
            for v in melody_col:
                j = v - min_melody_pitch
                if j >= self.instrument_range:
                    continue
                pattern_track[i, j + self.drum_range] = 1

        return pattern_track

    # does the same, but vice-versa
    # TODO -- check + test implementation
    def convert_numpy_image_to_pair(self, image: np.array) -> DrumMelodyPair:
        # Step 1. Fill in drum part
        drum_pattern = []
        for i in range(self.time_count):
            row = []
            for j in range(self.drum_range):
                row.append(image[i, j])
            drum_pattern.append(row)

        melody_pattern = []
        rand_min = random.randint(14) + 24# why not ? :)
        for i in range(self.time_count):
            row = []
            for j in range(self.drum_range, self.drum_range + self.instrument_range):
                row.append(image[i, j] + rand_min)
            melody_pattern.append(row)
        return DrumMelodyPair(drum_pattern, melody_pattern, 120, 1, 1)

# Эту функцию будем использовать для генерирования обучающей выборки
# Здесь же можно производить аугментацию обучающей выборки, к примеру
# В качестве аугментации можно использовать транспонирование
def make_numpy_dataset(img_size = (128, 50), limit = 1000):
    # read csv
    patterns_file = "patterns.pairs.tsv"
    dataset_with_melody = parse_csv(patterns_file, limit=limit)
    # initialize converter
    converter = Converter(img_size)
    # prepare numpy lists
    dataset_with_melody_np = []
    dataset_without_melody_np = []
    for img in dataset_with_melody:
        img_empty = DrumMelodyPair(img.drum_pattern, [], img.tempo, img.instrument, img.denominator)

        np_img_empty = converter.convert_pair_to_numpy_image(img_empty)
        np_img_dnb = converter.convert_pair_to_numpy_image(img)
        # add color channel
        np_img_empty = np.stack((np_img_empty,) * 1, axis=-1)
        np_img_dnb = np.stack((np_img_dnb,) * 1, axis=-1)
        dataset_without_melody_np.append(np_img_empty)
        dataset_with_melody_np.append(np_img_dnb)

    return np.array(dataset_with_melody_np), np.array(dataset_without_melody_np)


if __name__ == '__main__':
    pairs = parse_csv("test.tsv")
    #a = Converter((128, 50))
    a = Converter((256, 256))
    picts = []
    for i in pairs:
        picts.append(a.convert_pair_to_numpy_image(i))

