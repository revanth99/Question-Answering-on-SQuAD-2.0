# Question-Answering-on-SQuAD-2.0
The following code is the Squad 2.0 dataset. In the dataset, there are several Wikipidea articles, and each article in further separated into context paragraphs. For each context paragrpah, the dataset contains several questions, some of which are considered answerable. For each answerable question, the answer is a text span from the context paragraph. A model trained on the Squad 2.0 dataset should predict whether a given question is answerable or not, and if it is, it should predict the span from the context paragraph that answers the question.

We are using a Bi-Directional Attention Flow (BiDAF) model. the BiDAF includes character-level, word-level and contextual embeddings and uses the attention flow in both directions to get a query-aware context representation. 

# Code Organization

    ├── data_loader.py     <- Data is taken in batches and given to model.
    ├── eval.py            <- Evaluate the model with new context and question pair.
    ├── layers.py          <- Various layers are defined in BiDAF Architecture.
    ├── make_dataset.py    <- Load Dataset and train model.py
    ├── model.py.          <- Define the BiDAF model architecture.
    ├── test.py            <- Dev Dataset is used to test the trained model.
    ├── train.py           <- Train a model using the TRAIN dataset only.
    ├── utils.py           <- Collaborate the required functions to train the model. 
    
# Requirements

- 'Python 3.6'

# Performance Metrics
Exact Match(EM) and F1 Score are the performance metrics used to evaluate the model

# Set-Up

* Clone the repository
* Download GloVE word vectors: https://nlp.stanford.edu/projects/glove/
* Install the dependencies: ' python -m spacy download en'
* Run 'python make_dataset.py' to download SquAD dataset and pre-process the data
* Run 'python train.py' to train the model with hyper-parameters
* Run 'python test.py' to test the model EM and F1 scores on Dev examples
* Run 'python eval.py' to answer your own questions! 
