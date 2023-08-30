# taken from https://pytorch.org/tutorials/beginner/translation_transformer.html

# BSD 3-Clause License
#
# Copyright (c) 2017-2022, Pytorch contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import matplotlib.ticker as ticker
import numpy as np
from torchtext.data.metrics import bleu_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from architectures.Transformer import Seq2SeqTransformer
import torch.nn as nn
import torch
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
import datetime
import matplotlib.pyplot as plt

# Function to print to log file and console with timestamp
def printLog(log):
    with open("log_transformer.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {log}\n")
    print(log)


# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

# Select the source and target language
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'de'

# Place-holders
token_transform = {}
vocab_transform = {}

# Create source and target language tokenizer. Make sure to install the dependencies.
token_transform[SRC_LANGUAGE] = get_tokenizer(
    'spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer(
    'spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(
        split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE), root="data")
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

# Set the device to cuda if GPU is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# function to generate masks
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# Make PyTorch deterministic
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Set hyperparameters
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0

# Helper function to get an instance of the model with initialized parameters
def getModel():
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    return transformer

# Helper functions to get loss and optimizer
def getLoss(alpha: float):
    return torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=alpha)


def getOptimizer(parameters):
    return torch.optim.Adam(parameters, lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               # Numericalization
                                               vocab_transform[ln],
                                               tensor_transform)  # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# train the model for one epoch, return the loss
def train_epoch(model, optimizer, loss_fn):
    model.train()
    losses = 0
    train_iter = Multi30k(
        split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(
        train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

# Calculate validation loss
def evaluate(model, temp=1.0, alpha=0.0):
    model.eval()
    losses = 0
    loss_fn = getLoss(alpha)
    val_iter = Multi30k(split='valid', language_pair=(
        SRC_LANGUAGE, TGT_LANGUAGE))

    # bleu = calculate_bleu(val_iter, model)
    # print(f'BLEU score = {bleu*100:.2f}')
    val_dataloader = DataLoader(
        val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)/temp

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

# Calculate validation NLL
def calcNLL(model, temp=1.0):
    model.eval()
    losses = 0
    nll = torch.nn.NLLLoss(ignore_index=PAD_IDX)
    val_iter = Multi30k(split='valid', language_pair=(
        SRC_LANGUAGE, TGT_LANGUAGE))

    val_dataloader = DataLoader(
        val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)/temp
        logits = nn.functional.log_softmax(logits)
        tgt_out = tgt[1:, :]
        loss = nll(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, temp=1.0):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)/temp
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, temp=1.0):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, temp=temp).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

# calculate BLEU score
def calculate_bleu(data, model, temp=1.0):
    trgs = []
    pred_trgs = []
    for datum in data:
        src = datum[0]
        trg = datum[1]
        pred_trg = translate(model, src, temp=temp)
        pred_trgs.append(pred_trg.split())
        trgs.append([trg.split()])
    return bleu_score(pred_trgs, trgs)


# helper function to calulate the accuracy
def compute_logit(model, src, tgt):
    model.eval()
    with torch.no_grad():
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

    return logits


def t_compute_accuracy(model, data_loader, device, temperature=1.0):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
            logits = compute_logit(model, features, targets)/temperature
            _, predicted_labels = torch.max(logits, 2)
            targets = targets[1:].flatten()
            num_examples += targets.size(0)
            correct_pred += (predicted_labels.flatten() ==
                             targets).sum().float().mean()
    return correct_pred.float()/num_examples * 100

# helper function to calulate the reliability
def t_compute_reliability(model, data_loader, device, temperature=1.0):
    model.eval()
    correct_pred = []
    maxs = []
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
            logits = compute_logit(model, features, targets)/temperature
            targets = targets[1:]
            m = nn.Softmax(dim=2)
            probs = m(logits)
            max, predicted_labels = torch.max(probs, 2)
            correct_pred.append(
                (predicted_labels == targets).float().flatten())
            maxs.append(max.float().flatten())
    correct_pred = torch.concat(correct_pred)
    maxs = torch.concat(maxs)
    return correct_pred, maxs

# Internal helper function
def t_bin_reliability(model, validation_loader, device, num_bins, temperature=1.0):
    a, c = t_compute_reliability(model, validation_loader, device, temperature)

    reliability = torch.zeros((3, num_bins))
    a = a.to(reliability.device)
    c = c.to(reliability.device)

    for i, elem in enumerate(c):
        reliability[2, min(math.floor(elem*num_bins), num_bins-1)] += elem
        reliability[1, min(math.floor(elem*num_bins), num_bins-1)] += 1
        reliability[0, min(math.floor(elem*num_bins), num_bins-1)] += int(a[i])
    reliability[2, :] /= reliability[1, :]
    return reliability

# Create a plots of the BLEU score and ECE for different temperatures
def bleuECEPlot(temps, bleus, eces, alpha=0.0, ax1=None):
    marker = '+' if alpha != 0 else None
    ax1.title.set_text(rf"$\alpha = {alpha}$")
    color = 'tab:blue'
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('BLEU Score')
    ax1.plot(temps, bleus, color=color, marker=marker)
    ax1.tick_params(axis='y')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('ECE')  # we already handled the x-label with ax1
    ax2.plot(temps, eces, color=color, marker=marker)
    ax2.tick_params(axis='y')
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

# Create a plots of the NLL for different temperatures
def NLLPlot(temps, NLLs, NLLsSmooth, ax1):
    marker = '+'
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('NLL')
    ax1.plot(temps, NLLs)
    ax1.plot(temps, NLLsSmooth, marker=marker)
    ax1.tick_params(axis='y')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

# Function for calculating the ECE
def calculate_ece(reliability_bins):
    n = torch.sum(reliability_bins[1])
    sum = 0
    for i, elem in enumerate(reliability_bins[0]):
        if not torch.isnan(reliability_bins[2, i]):
            sum += (reliability_bins[1, i]/n)*torch.abs(
                (reliability_bins[0, i] / reliability_bins[1, i])-reliability_bins[2, i])
    return sum

# Run the training for a given number of epochs and alpha, save the model and calculate the BLEU score
def runTraining(epochs=18, alpha=0.0):
    printLog(f"Running Training for {epochs} epochs with alpha={alpha}")
    from timeit import default_timer as timer
    transformer = getModel()
    optimizer = getOptimizer(transformer.parameters())
    loss_fn = getLoss(alpha=alpha)
    for epoch in range(1, epochs+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, alpha=alpha)
        printLog(
            (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save(transformer.state_dict(),
               f"models/Transformer_Multi30K_IMC/Transformer_alpha={alpha}.pt")
    test_data = Multi30k(split='valid', language_pair=(
        SRC_LANGUAGE, TGT_LANGUAGE))
    bleu = calculate_bleu(test_data, transformer)
    printLog(f'BLEU score = {bleu*100:.2f}')

# Create a reliability plot for a given alpha and temperature
def runReliability(alpha, temperature, show=False):
    from util.common import plot_reliability
    printLog(
        f"Running reliability with alpha={alpha} and temperature={temperature}")
    transformer = getModel()
    device = DEVICE
    num_bins = 15
    val_iter = Multi30k(split='valid', language_pair=(
        SRC_LANGUAGE, TGT_LANGUAGE))
    validation_loader = DataLoader(
        val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    transformer.load_state_dict(torch.load(
        f"models/Transformer_Multi30K_IMC/Transformer_alpha={0.0}.pt", map_location=device))
    printLog(
        f"Accuarcy for Transformer w. Hard Labels: {t_compute_accuracy(transformer, validation_loader, device).item():.2f}%")
    rel_bins_Hard = t_bin_reliability(
        transformer, validation_loader, device, num_bins)
    rel_bins_Temperature = t_bin_reliability(
        transformer, validation_loader, device, num_bins, temperature)
    transformer.load_state_dict(torch.load(
        f"models/Transformer_Multi30K_IMC/Transformer_alpha={alpha}.pt", map_location=device))
    printLog(
        f"Accuarcy for Transformer w. Smooth Labels (alpha={alpha}): {t_compute_accuracy(transformer, validation_loader, device).item():.2f}%")
    rel_bins_Smooth = t_bin_reliability(
        transformer, validation_loader, device, num_bins)

    fig = plot_reliability(rel_bins_Hard, rel_bins_Smooth,
                           rel_bins_Temperature, temperature, alpha)
    fig.savefig(
        format="pdf", fname=f"figures/transformer_reliability_{alpha}.pdf", bbox_inches='tight')
    eceHL = calculate_ece(rel_bins_Hard)
    eceSL = calculate_ece(rel_bins_Smooth)
    eceTS = calculate_ece(rel_bins_Temperature)
    printLog(f"ECE w. Hard Labels: {eceHL:.4f}")
    printLog(f"ECE w. Smooth Labels (alpha={alpha}): {eceSL:.4f}")
    printLog(
        f"ECE w. Temperature Scaling (temperature={temperature}): {eceTS:.4f}")
    if show:
        fig.show()

# Create plots for BLEU, ECE and NLL depending on the temperature for different alphas
def runCalibrationEffects(alphas: list(), temps, show=False):
    printLog(
        f"Running calibrationEffects with alphas={alphas} and temps={temps}")
    val_iter = Multi30k(split='valid', language_pair=(
        SRC_LANGUAGE, TGT_LANGUAGE))
    validation_loader = DataLoader(
        val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    bleus = []
    device = DEVICE
    num_bins = 15
    eces = []
    nlls = []
    bleusSmooth = {}
    ecesSmooth = {}
    nllsSmooth = {}
    transformer = getModel()
    transformer.load_state_dict(torch.load(
        f"models/Transformer_Multi30K_IMC/Transformer_alpha={0.0}.pt", map_location=device))
    for i in range(temps.shape[0]):
        bleus.append(calculate_bleu(val_iter, transformer, temp=temps[i])*100)
        rel_bins = t_bin_reliability(
            transformer, validation_loader, device, num_bins, temps[i])
        eces.append(calculate_ece(rel_bins))
        nlls.append(calcNLL(transformer, temp=temps[i]))
    for alpha in alphas:
        plt.rcParams['text.usetex'] = True
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = 13
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        transformer.load_state_dict(torch.load(
            f"models/Transformer_Multi30K_IMC/Transformer_alpha={alpha}.pt", map_location=device))
        bleusSmooth[alpha] = []
        ecesSmooth[alpha] = []
        nllsSmooth[alpha] = []
        for i in range(temps.shape[0]):
            bleusSmooth[alpha].append(calculate_bleu(
                val_iter, transformer, temp=temps[i])*100)
            rel_bins = t_bin_reliability(
                transformer, validation_loader, device, num_bins, temps[i])
            ecesSmooth[alpha].append(calculate_ece(rel_bins))
            nllsSmooth[alpha].append(calcNLL(transformer, temp=temps[i]))
        bleuECEPlot(temps, bleus, eces, 0.0, ax1)
        bleuECEPlot(temps, bleusSmooth[alpha], ecesSmooth[alpha], alpha, ax2)
        NLLPlot(temps, nlls, nllsSmooth[alpha], ax3)
        fig.savefig(
            format="pdf", fname=f"figures/transformer_calibrationEffects_{alpha}.pdf", bbox_inches='tight')
    if show:
        fig.show()
