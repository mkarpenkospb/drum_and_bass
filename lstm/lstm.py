import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

from decode_patterns import data_conversion

torch.manual_seed(1)

lstm = nn.LSTM(14, 36)  # Input dim is 14, output dim is 36
inputs = [torch.randn(1, 14) for _ in range(32)]  # make a sequence of length 32

# initialize the hidden state.
# hidden = (torch.randn(1, 1, 36),
#           torch.randn(1, 1, 36))
# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 36), torch.randn(1, 1, 36))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)


# define data loader
# import dataset
drum, bass = data_conversion.make_lstm_dataset(height=128, limit=250)
# take 80% to training, other to testing
L = len(drum)
idx = np.arange(L) < 0.8*L
np.random.shuffle(idx)
drum_train = drum[idx]
bass_train = bass[idx]
drum_test = drum[np.logical_not(idx)]
bass_test = bass[np.logical_not(idx)]

def prepare_sequence(seq):
    return tuple(torch.tensor(seq[i]) for i in range(seq.shape[0]))

class LSTMBassSequencer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, octave_size):
        super(LSTMBassSequencer, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes drum lines as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to bass space

        self.hidden2bass = nn.Linear(hidden_dim, octave_size)

    def forward(self, drum_line):
        inputs = torch.cat(drum_line).view(len(drum_line), 1, -1)
        lstm_out, lstm_hidden = self.lstm(inputs)

        tag_space = self.hidden2bass(lstm_out.view(len(drum_line), -1))
        tag_scores = torch.sigmoid(tag_space) #F.sigmoid
        return tag_scores

model = LSTMBassSequencer(14, 36, 36)
loss_function = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
def print_test_result():
    with torch.no_grad():
        inputs = prepare_sequence(drum_test[0])
        bass_from_drum = model(inputs)
        print(bass_from_drum)

print_test_result()

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    if (epoch % 10 == 0):
        print()
        print_test_result()
        print(f"Epoch #{epoch+1}.", end="")
    else:
        print(".", end="")
    for drum, bass in zip(drum_train, bass_train):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        drum_seq_in = prepare_sequence(drum)
        bass_seq_target = prepare_sequence(bass)

        # Step 3. Run our forward pass.
        bass_result = model(drum_seq_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(bass_result, torch.tensor(bass))
        loss.backward()
        optimizer.step()

# See what the scores are after training
print_test_result()