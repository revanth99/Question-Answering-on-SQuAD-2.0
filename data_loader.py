import torch.utils.data as data


class SquadDataset(data.Dataset):

    def __init__(self, w_context, c_context, w_question, c_question, labels):
        self.w_context = w_context
        self.c_context = c_context
        self.w_question = w_question
        self.c_question = c_question
        self.labels = labels

    def __getitem__(self, index):
        return self.w_context[index], self.c_context[index], self.w_question[index], self.c_question[index],\
               self.labels[index]

    def __len__(self):
        return len(self.w_context)
