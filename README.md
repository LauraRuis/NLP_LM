# NLP_LM
NLP project

## Baseline kenlm
Before usage run `pip install kenlm`

To check perplexity of baseline kenlm 3gram, 4gram or 5gram, change `5gram.arpa` in python file baseline_kenlm.py and run `> python baseline_kenlm.py`

## Baseline Bengio
Change training parameters in python file train_NLM.py (e.g. cuda, epochs, embedding_dim, context_size, this last one is now 2 (giving a 3-gram))

Run `> python train_NLM.py` for training

## RAN
Change training parameters in python file train_RAN.py (e.g. cuda, epochs, embedding_dim, hidden_size, tanh_act)

Run `> python train_RAN.py` for training

NB. if tanh_act is set to True, the tanh activation function will be used giving the more complex RAN instead of the linear one, when using the linear model with tanh_act set to False, the number of hidden nodes and the embedding_dim should be equal.

