import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from model import BiDAF
from data_loader import SquadDataset
from utils import compute_batch_metrics

data_dir = "/Users/revan/PycharmProjects/Question-Answering-based-on-SQuAD-master/Question-Answering-based-on-SQuAD-master/SQuAD/"
train_dir = data_dir + "train/"
dev_dir = data_dir + "dev/"

# preprocessing values used for training
prepro_params = {
    "max_words": -1,
    "word_embedding_size": 100,
    "char_embedding_size": 100,
    "max_len_context": 400,
    "max_len_question": 50,
    "max_len_word": 25
}

# hyper-parameters setup
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

writer = SummaryWriter(experiment_path)

dev_features = np.load(os.path.join(dev_dir, "dev_features.npz"), allow_pickle=True)
d_w_context, d_c_context, d_w_question, d_c_question, d_labels = dev_features["context_idxs"],\
                                                                 dev_features["context_char_idxs"],\
                                                                 dev_features["question_idxs"],\
                                                                 dev_features["question_char_idxs"],\
                                                                 dev_features["label"]


with open(os.path.join(train_dir, "word2idx.pkl"), "rb") as f:
    word2idx = pickle.load(f)

idx2word = dict([(y, x) for x, y in word2idx.items()])

with open(os.path.join(train_dir, "word_embeddings.pkl"), "rb") as e:
    word_embedding_matrix = pickle.load(e)
with open(os.path.join(train_dir, "char_embeddings.pkl"), "rb") as e:
    char_embedding_matrix = pickle.load(e)


word_embedding_matrix = torch.from_numpy(np.array(word_embedding_matrix)).type(torch.float32)
char_embedding_matrix = torch.from_numpy(np.array(char_embedding_matrix)).type(torch.float32)

# load dataset
test_dataset = SquadDataset(d_w_context, d_c_context, d_w_question, d_c_question, d_labels)

# load data generator
test_dataloader = DataLoader(test_dataset,
                             shuffle=True,
                             batch_size=hyper_params["batch_size"],
                             num_workers=4)

print("Length of test data loader is:", len(test_dataloader))

# load the model
model = BiDAF(word_vectors=word_embedding_matrix,
              char_vectors=char_embedding_matrix,
              hidden_size=hyper_params["hidden_size"],
              drop_prob=hyper_params["drop_prob"])
try:
    model.load_state_dict(torch.load(os.path.join("output/exp-1", "model_final.pkl"),
                                         map_location=lambda storage, loc: storage)["state_dict"])
    print("Model weights successfully loaded.")
except:
    print("Model weights not found, initialized model with random weights.")

# define loss criterion
criterion = nn.CrossEntropyLoss()

model.eval()
test_em = 0
test_f1 = 0
n_samples = 0
with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        w_context, c_context, w_question, c_question, labels = batch[0].long().to(device),\
                                                                     batch[1].long().to(device),\
                                                                     batch[2].long().to(device),\
                                                                     batch[3].long().to(device),\
                                                                     batch[4]
        pred1, pred2 = model(w_context, c_context, w_question, c_question)
        em, f1 = compute_batch_metrics(w_context, idx2word, pred1, pred2, labels)
        test_em += em
        test_f1 += f1
        n_samples += w_context.size(0)

    writer.add_scalars("test", {"EM": np.round(test_em / n_samples, 2),
                                "F1": np.round(test_f1 / n_samples, 2)})
    print("Test EM of the model after training is: {}".format(np.round(test_em / n_samples, 2)))
    print("Test F1 of the model after training is: {}".format(np.round(test_f1 / n_samples, 2)))

