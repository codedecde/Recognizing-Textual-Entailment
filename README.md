# Recognizing-Textual-Entailment
---------------------
A pyTorch implementation of models used for Recognizing Textual Entailment using the [SNLI](https://nlp.stanford.edu/projects/snli/) corpus. The following models have been implemented (so far) :

* Reasoning About Entailment With Neural Attention ( [Rocktaschel et. al. '15](https://arxiv.org/pdf/1509.06664.pdf) )
* Learning Natural Language Inference with LSTM ( [Wang & Jiang '15](http://www.aclweb.org/anthology/N16-1170))

The details and results specific to the different models are given below
### Reasoning About Entailment with Neural Attention
---------------------
#### Introduction
----------------------------
The paper presents an LSTM based model with attention for the task. The following are some key points:

* Two LSTM's encode the premise and hypothesis.
* The hidden state of the LSTM encoding the hypothesis is initialised using the hidden state of the LSTM encoding the premise
* Two different attention mechanisms are explored:
	* Using just the last output of the hypothesis LSTM to attend over the outputs of the premise LSTM
	* Attending over the premise LSTM outputs at every step of processing the hypothesis (using a simple RNN).
#### Running the code
--------------------------------
To start training the model, call
```bash
python run_rte.py 
``` 
The following command line arguments are available:
General arguments (used by other models as well)


    -n_embed     (Embedding Layer Dimensions, default 300)
    -n_dim       (Hidden Layer Dimensions, default 300)
    -batch	     (batch size, default 256)
    -dropout	 (p value for dropout layer, default 0.1)
    -l2			 (L2 regularisation value, default 0.0003)
    -lr		     (Learning rate, default 0.001 )
    -last_nonlinear	(Projection to softmax layer is non-linear or not, default False)
    -train_flag  (Training or evaluation mode, default True)
Model specific arguments    
	
	-wbw_attn    (Use word by word attention, default False)
    -h_maxlen    (Maximum Length of hypothesis(used by the recurrent batchnorm layer), default 30)


#### Implementation Caveats
--------------------------

The word by word attention model is basically a simple RNN, used to attend over the premise at every step. Consequently, it faces the exploding gradient problem. In order to prevent that from happening, the following measures have been taken:

* Setting the initial weights of the RNN to be orthogonal
* Using Batch Normalisation in Recurrent Networks, as done in Recurrent Batch Normalisation [Cooijmans et. al. '17](https://arxiv.org/pdf/1603.09025.pdf) [see **recurrent_BatchNorm.py** for implementation.]

#### Results 
---------------------------





