import mido
import csv
import numpy as np
from mido import MidiFile, MidiTrack, Message

port = mido.open_output(mido.get_output_names()[0])

mid = mido.MidiFile("C:\\Users\\Asus\\Desktop\\Masters\\Проект\\файлы\\midi raw\\1208.mid")
# for msg in mid.play():
#     port.send(msg)
for msg in mid:
    print(msg)
##########раскомментировать нужную часть

# patterns_file = "C:\\Users\\Asus\\Desktop\\Masters\\Проект\\файлы\\patterns.pairs.tsv"
# cnt = 0
#
# midi = []
# midi_tempo = []
#
#
# max_pattern = []
#
# ml_drum = 0
#
# ALLOWED_PITCH = np.array(list(reversed([36, 38, 42, 41, 45, 46, 48, 49, 51, 58, 60, 61, 62, 64])))
#
#
# a = np.array([1,3,2,4,5])
# a[np.array([True,False,True,False,False])]
# set(np.array([1, 2]))
#
#
# with open(patterns_file) as csvfile:
#     spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
#     for row in spamreader:
#         pattern = [np.array([bool(int(i)) for i in j]) for j in
#                        [bin(int(x))[2:].rjust(14, '0') for x in row[10].split(',')]
#                     ]
#         midi_tempo.append(int(row[3]))
#         #pattern = [bin(int(x))[2:].rjust(14, '0') for x in row[10].split(',')]
#         drum_inst = [int(x) for x in row[5].split(',')]
#         if len(drum_inst) > ml_drum:
#             ml_drum = len(drum_inst)
#             max_pattern = drum_inst
#         time_instr = [list(ALLOWED_PITCH[i]) for i in pattern]
#         print(time_instr)
#         midi.append(time_instr)
#         cnt += 1
#         if cnt > 10:
#             break
#
# print(midi)
#
# mid = MidiFile()
# track = MidiTrack()
# mid.tracks.append(track)
#
# time = 0;
#
# tr_num = 8
# time_int = midi_tempo[tr_num]
#
# for i in (midi[tr_num]):
#     if not i:
#         time += time_int
#         continue
#     else:
#         track.append(Message('note_on', note=i[0], velocity=127, time=time, channel=9))
#     for j in i[1:]:
#         track.append(Message('note_on', note=j, velocity=127, time=0, channel=9))
#     time = time_int
#
#
# for msg in mid.play():
#     port.send(msg)



if __name__ == '__main__':
    pass
