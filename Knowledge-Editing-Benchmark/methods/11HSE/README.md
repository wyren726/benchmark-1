# HSE
- Code for [``Hippocampal-like Sequential Editing for Continually Knowledge Update in Large Language Models``]


## Requirements
**At least one A40 48G GPU.**

- pytorch==1.12.1
- einops==0.4.0
- higher==0.2.1
- hydra-core==1.2.0
- transformers==4.23.1
- datasets==1.18.3
- matplotlib==3.6.1
- spacy==3.4.1
- scipy==1.9.2
- scikit-learn==1.0.2
- nltk==3.7

## Quick Start
### An example for editing Llama3 (8B) on counterfact dataset
#### 1. Edit Llama3 (8B) model 
 
    python3 -m experiments.evaluate     --alg_name=HSE     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B.json --ds_name=mcf --dataset_size_limit=1000    --num_edits=1 --downstream_eval_steps=100

This command runs an evaluation script for the HSE algorithm using the Llama3-8b-instruct. Below are the explanations for each argument:

- `--alg_name=HSE`: Specifies the name of the algorithm being used, which is HSE in this case.
- `--model_name=meta-llama/Meta-Llama-3-8B-Instruct`: Indicates the name of the model being evaluated, here it is Llama-3-8B-Instruct.
- `--hparams_fname=Llama3-8B.json`: Points to the JSON file containing hyperparameters specific to the Llama-3-8B-Instruct model.
- `--ds_name=mcf`: Specifies the dataset name, in this case, "mcf".
- `--dataset_size_limit=1000`: Sets the total number of editing samples to 1000.
- `--num_edits=1`: Defines the batch size for each round of editing, meaning 1 edit will be performed in each batch. 
- `--downstream_eval_steps=100`: indicates that a test of general capabilities is conducted after every 100 rounds of editing.
#### 2. Summarize the results

    python3 -m experiments.summarize --dir_name=HSE --runs=run_***

## Acknowledgment
Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git) and [``ALPHAEDIT``](https://github.com/jianghoucheng/AlphaEdit). 
