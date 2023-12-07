import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import spacy
import random

import argparse

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--seed', required=True, help='seed')
parser.add_argument('--init_method', required=True, help='init_method')

args = parser.parse_args()


SEED = int(args.seed)
INIT_METHOD = int(args.init_method)

print('seed :', SEED)
print('init_method :', INIT_METHOD)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text):
    return [token.text for token in spacy_en.tokenizer(text)]

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en',
                  batch_first = True)

LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples   : {len(train_data)}')
print(f'Number of validation examples : {len(valid_data)}')
print(f'Number of testing examples    : {len(test_data)}')

MAX_VOCAB_SIZE = 20000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary : {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

print(TEXT.vocab.freqs.most_common(20))

print(TEXT.vocab.itos[:10])

print(LABEL.vocab.stoi)
BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

print('Number of minibatch for training dataset   : {}'.format(len(train_iterator)))
print('Number of minibatch for validation dataset : {}'.format(len(valid_iterator)))
print('Number of minibatch for testing dataset    : {}'.format(len(test_iterator)))


embedding_dim = 256
hidden_units = 128
EPOCHS = 50
learning_rate = 5e-4

class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, embedding_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, _ = self.rnn(embedded)
        output = self.linear(output[:, -1, :])
        return output

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

model = LSTM(len(TEXT.vocab), 128, len(LABEL.vocab)-1, 300, 0.2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()


model = model.to(device)
if INIT_METHOD == 0:
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    model.apply(initialize_weights)
    print('nn.init.xavier_uniform_(m.weight.data)')

# elif INIT_METHOD == 1:
#     def initialize_weights(m):
#         if hasattr(m, 'weight') and m.weight.dim() > 1:
#             nn.init.xavier_normal_(m.weight.data)
#     model.apply(initialize_weights)
#     print('nn.init.xavier_normal_(m.weight.data)')

elif INIT_METHOD == 2:
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    model.apply(initialize_weights)
    print('nn.init.kaiming_uniform_(m.weight.data, mode=fan_in, nonlinearity=relu')

# elif INIT_METHOD == 3:
#     def initialize_weights(m):
#         if hasattr(m, 'weight') and m.weight.dim() > 1:
#             nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
#     model.apply(initialize_weights)
#     print('nn.init.kaiming_normal_(m.weight.data, mode=fan_out, nonlinearity=relu)')

# elif INIT_METHOD == 4:
#     def initialize_weights(m):
#         if hasattr(m, 'weight') and m.weight.dim() > 1:
#             nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
#     model.apply(initialize_weights)
#     print('nn.init.kaiming_uniform_(m.weight.data, mode=fan_in), nonlinearity=leaky_relu)')

# elif INIT_METHOD == 5:
#     def initialize_weights(m):
#         if hasattr(m, 'weight') and m.weight.dim() > 1:
#             nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')
#     model.apply(initialize_weights)
#     print('nn.init.kaiming_normal_(m.weight.data, mode=fan_out, nonlinearity=leaky_relu)')


# elif INIT_METHOD == 6:
#     def initialize_weights(m):
#         if hasattr(m, 'weight') and m.weight.dim() > 1:
#             nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity='relu')
#     model.apply(initialize_weights)
#     print('nn.init.kaiming_uniform_(m.weight.data, mode=fan_out, nonlinearity=leaky_relu)')

elif INIT_METHOD == 7:
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    model.apply(initialize_weights)
    print('nn.init.kaiming_normal_(m.weight.data, mode=fan_in, nonlinearity=relu)')


criterion = criterion.to(device)

def binary_accuracy(preds, target):
    '''
    from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
    '''
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == target).float()

    acc = correct.sum() / len(correct)
    return acc



def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        # We initialize the gradient to 0 for every batch.
        optimizer.zero_grad()

        # batch of sentences인 batch.text를 model에 입력
        predictions = model(batch.text).squeeze(1)

        # Calculate the loss value by comparing the prediction result with batch.label 
        loss = criterion(predictions, batch.label)

        # Accuracy calculation
        acc = binary_accuracy(predictions, batch.label)

        # Backpropagation using backward()
        loss.backward()

        # Update the parameters using the optimization algorithm
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    # "evaluation mode" : turn off "dropout" or "batch nomalizaation"
    model.eval()

    # Use less memory and speed up computation by preventing gradients from being computed in pytorch
    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time
import pickle

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')

log = {'train_acc': [], 'train_loss':[], 'val_acc':[], 'val_loss':[], 'test_acc':[], 'test_loss':[]}
for epoch in range(EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    log['train_loss'].append(train_loss)
    log['train_acc'].append(train_acc)
    log['val_loss'].append(valid_loss)
    log['val_acc'].append(valid_acc)

model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)
log['test_loss'].append(test_loss)
log['test_acc'].append(test_acc)


with open(f'lstm_method_1_{INIT_METHOD}_{SEED}.pkl', 'wb') as tf:
    pickle.dump(log, tf)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')