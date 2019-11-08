import mido
import csv
import numpy as np
from mido import MidiFile, MidiTrack, Message, bpm2tempo
from collections import defaultdict

port = mido.open_output(mido.get_output_names()[0])

# mid = mido.MidiFile("C:\\Users\\Asus\\Desktop\\Masters\\Проект\\файлы\\midi raw\\1208.mid")
# for msg in mid.play():
#     port.send(msg)
# for msg in mid:
#     print(msg)
##########раскомментировать нужную часть

patterns_file = "test.tsv"

base_for_mel = [[] for i in range(32)]

cnt = 0
ml_drum = 0
midi = []
midi_tempo = []
max_pattern = []
melody = []
melody_instr = []
temp_denomts = []
ALLOWED_PITCH = np.array(list(reversed([36, 38, 42, 41, 45, 46, 48, 49, 51, 58, 60, 61, 62, 64])))
with open(patterns_file) as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        pattern = [np.array([bool(int(i)) for i in j]) for j in
                       [bin(int(x))[2:].rjust(14, '0') for x in row[10].split(',')]
                    ]
        midi_tempo.append(int(row[3]))
        drum_inst = [int(x) for x in row[5].split(',')]
        midi_tempo.append(int(row[3]))
        time_instr = [list(ALLOWED_PITCH[i]) for i in pattern]
        #добавим инструмент, котторым играется мелодия
        melody_instr.append(int(row[13]))
        pairs = eval(row[15])
        #обработка мелодии
        temp_den = int(row[8])
        temp_denomts.append(temp_den)
        mel = [[] for i in range(32 * temp_den)]
        for i in pairs:
            if i[0] >= 32 * temp_den:
                continue
            mel[i[0]].append(i[1])

        melody.append(mel)
        #print(time_instr)
        print(len(pairs), ",  ", pairs)
        midi.append(time_instr)
        cnt += 1
        if cnt > 10:
            break

print("\n\ndrum\n\n")

for i in midi:
    print(i)


print("\n\nmelodies\n\n")
for i in melody:
    print(i)

mid = MidiFile(type=1)
track = MidiTrack()
mid.tracks.append(track)

time = 0
tr_num = 6
time_int = 0
k = 1

def adjust_tempo():
    bpm = midi_tempo[tr_num]
    bpm_default = 120
    global k, time_int
    k = bpm_default / bpm
    time_int = int((480 / 4) * k)

adjust_tempo()

repeat = 4
while repeat > 0:
    repeat -= 1
    for i in (midi[tr_num]):
        if not i:
            time += time_int
            continue

        for j in i:
            track.append(Message('note_on', note=j, velocity=127, time=int(time), channel=9))
            time = 0
        time = time_int

track2 = MidiTrack()
mid.tracks.append(track2)
track2.append(Message('program_change', program=melody_instr[tr_num], time=0, channel=2))

time = 0
repeat = 4
last_notes = []
while repeat > 0:
    repeat -= 1
    for i in (melody[tr_num]):
        if not i:
            time += time_int/temp_denomts[tr_num]
            continue

        for j in last_notes:
            track2.append(Message('note_off', note=j, velocity=127, time=int(time), channel=2))
            time = 0
        for j in i:
            track2.append(Message('note_on', note=j, velocity=127, time=int(time), channel=2))
            time = 0
        time = time_int/temp_denomts[tr_num]
        last_notes = i

for j in last_notes:
    track2.append(Message('note_off', note=j, velocity=127, time=int(time), channel=2))
    time = 0

mid.save("../midi/sample3.mid")


for msg in mid.play():
    port.send(msg)

if __name__ == '__main__':
    pass
