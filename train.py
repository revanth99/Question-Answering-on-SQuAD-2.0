import numpy as np
import pickle
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from model import BiDAF
from data_loader import SquadDataset
from utils import save_checkpoint, compute_batch_metrics

prepro_params = {
    "max_words": -1,
    "word_embedding_size": 100,
    "char_embedding_size": 8,
    "max_len_context": 100,
    "max_len_question": 50,
    "max_len_word": 25
}

hyper_params = {
    "num_epochs": 1,
    "batch_size": 1,
    "learning_rate": .5,
    "hidden_size": 100,
    "char_channel_width": 5,
    "char_channel_size": 100,
    "drop_prob": 0.9,
    "pretrained": False
}

experiment_params = {"preprocessing": prepro_params, "model": hyper_params}




experiment_path = "output/{}".format("exp-1")
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)


with open(os.path.join(experiment_path, "config_{}.json".format("exp-1")), "w") as f:
    json.dump(experiment_params, f)


writer = SummaryWriter(experiment_path)

train_features = np.load(os.path.join(train_dir, "train_features.npz"), allow_pickle=True)
t_w_context, t_c_context, t_w_question, t_c_question, t_labels = train_features["context_idxs"],\
                                                                 train_features["context_char_idxs"],\
                                                                 train_features["question_idxs"],\
                                                                 train_features["question_char_idxs"],\
                                                                 train_features["label"]

dev_features = np.load(os.path.join(dev_dir, "dev_features.npz"), allow_pickle=True)
d_w_context, d_c_context, d_w_question, d_c_question, d_labels = dev_features["context_idxs"],\
                                                                 dev_features["context_char_idxs"],\
                                                                 dev_features["question_idxs"],\
                                                                 dev_features["question_char_idxs"],\
                                                                 dev_features["label"]


with open(os.path.join(train_dir, "word_embeddings.pkl"), "rb") as e:
    word_embedding_matrix = pickle.load(e)
with open(os.path.join(train_dir, "char_embeddings.pkl"), "rb") as e:
    char_embedding_matrix = pickle.load(e)


with open(os.path.join(train_dir, "word2idx.pkl"), "rb") as f:
    word2idx = pickle.load(f)

idx2word = dict([(y, x) for x, y in word2idx.items()])

word_embedding_matrix = torch.from_numpy(np.array(word_embedding_matrix)).type(torch.float32)
char_embedding_matrix = torch.from_numpy(np.array(char_embedding_matrix)).type(torch.float32)


train_dataset = SquadDataset(t_w_context, t_c_context, t_w_question, t_c_question, t_labels)
valid_dataset = SquadDataset(d_w_context, d_c_context, d_w_question, d_c_question, d_labels)


train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=hyper_params["batch_size"],
                              num_workers=4)

valid_dataloader = DataLoader(valid_dataset,
                              shuffle=True,
                              batch_size=hyper_params["batch_size"],
                              num_workers=4)

print("Length of training data loader is:", len(train_dataloader))
print("Length of valid data loader is:", len(valid_dataloader))

model = BiDAF(word_vectors=word_embedding_matrix,
              char_vectors=char_embedding_matrix,
              hidden_size=hyper_params["hidden_size"],
              drop_prob=hyper_params["drop_prob"])
if hyper_params["pretrained"]:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), hyper_params["learning_rate"], weight_decay=1e-4)

if hyper_params["pretrained"]:
    best_valid_loss = torch.load(os.path.join(experiment_path, "model.pkl"))["best_valid_loss"]
    epoch_checkpoint = torch.load(os.path.join(experiment_path, "model_last_checkpoint.pkl"))["epoch"]
    print("Best validation loss obtained after {} epochs is: {}".format(epoch_checkpoint, best_valid_loss))
else:
    best_valid_loss = 100
    epoch_checkpoint = 0

print("Starting training...")
for epoch in range(hyper_params["num_epochs"]):
    print("##### epoch {:2d}".format(epoch + 1))
    model.train()
    train_losses = 0
    for i, batch in enumerate(train_dataloader):
        w_context, c_context, w_question, c_question, label1, label2 = batch[0].long().to(device),\
                                                                       batch[1].long().to(device), \
                                                                       batch[2].long().to(device), \
                                                                       batch[3].long().to(device), \
                                                                       batch[4][:, 0].long().to(device),\
                                                                       batch[4][:, 1].long().to(device)
        optimizer.zero_grad()
        pred1, pred2 = model(w_context, c_context, w_question, c_question)
        loss = criterion(pred1, label1) + criterion(pred2, label2)
        train_losses += loss.item()

        loss.backward()
        optimizer.step()

    writer.add_scalars("train", {"loss": np.round(train_losses / len(train_dataloader), 2),
                                 "epoch": epoch + 1})
    print("Train loss of the model at epoch {} is: {}".format(epoch + 1, np.round(train_losses /
                                                                                  len(train_dataloader), 2)))

    model.eval()
    valid_losses = 0
    valid_em = 0
    valid_f1 = 0
    n_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            w_context, c_context, w_question, c_question, labels = batch[0].long().to(device), \
                                                                   batch[1].long().to(device), \
                                                                   batch[2].long().to(device), \
                                                                   batch[3].long().to(device), \
                                                                   batch[4]

            first_labels = torch.tensor([[int(a) for a in l.split("|")[0].split(" ")]
                                         for l in labels], dtype=torch.int64).to(device)
            pred1, pred2 = model(w_context, c_context, w_question, c_question)
            loss = criterion(pred1, first_labels[:, 0]) + criterion(pred2, first_labels[:, 1])
            valid_losses += loss.item()
            em, f1 = compute_batch_metrics(w_context, idx2word, pred1, pred2, labels)
            valid_em += em
            valid_f1 += f1
            n_samples += w_context.size(0)

        writer.add_scalars("valid", {"loss": np.round(valid_losses / len(valid_dataloader), 2),
                                     "EM": np.round(valid_em / n_samples, 2),
                                     "F1": np.round(valid_f1 / n_samples, 2),
                                     "epoch": epoch + 1})
        print("Valid loss of the model at epoch {} is: {}".format(epoch + 1, np.round(valid_losses /
                                                                                      len(valid_dataloader), 2)))
        print("Valid EM of the model at epoch {} is: {}".format(epoch + 1, np.round(valid_em / n_samples, 2)))
        print("Valid F1 of the model at epoch {} is: {}".format(epoch + 1, np.round(valid_f1 / n_samples, 2)))

    save_checkpoint({
        "epoch": epoch + 1 + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": np.round(valid_losses / len(valid_dataloader), 2)
    }, True, os.path.join(experiment_path, "model_last_checkpoint.pkl"))


    is_best = bool(np.round(valid_losses / len(valid_dataloader), 2) < best_valid_loss)
    best_valid_loss = min(np.round(valid_losses / len(valid_dataloader), 2), best_valid_loss)
    save_checkpoint({
        "epoch": epoch + 1 + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": best_valid_loss
    }, is_best, os.path.join(experiment_path, "model.pkl"))


writer.export_scalars_to_json(os.path.join(experiment_path, "all_scalars.json"))
writer.close()
