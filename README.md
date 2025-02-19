# LLM4Opt:
We introduce $\mathtt{MuMOInstruct}$, the first high-quality instruction-tuning dataset 
specifically focused on complex multi-property molecule optimization tasks. 
Leveraging $\mathtt{MuMOInstruct}$, we develop '$\mathtt{GeLLM^3O}$'s, a series of instruction-tuned LLMs for molecule optimization.
Extensive evaluations across 5 in-domain and 5 out-of-domain
tasks demonstrate that '$\mathtt{GeLLM^3O}$'s consistently outperform state-of-the-art baselines. 
'$\mathtt{GeLLM^3O}$'s also exhibit outstanding zero-shot generalization to unseen tasks, significantly outperforming powerful closed-source LLMs.
Such strong generalizability demonstrates the tremendous potential of '$\mathtt{GeLLM^3O}$'s as foundational models for molecule optimization,
thereby tackling novel optimization tasks without resource-intensive retraining. 

## Requirements

Please use `pip install -r requirements.txt` to install dependies. Ensure you have `python >= 3.10.0` installed.


## Dataset

The instruction-tuning dataset $\mathtt{MuMOInstruct}$ is available on [HuggingFace](). 

## Models

The instruction-tuned model checkpoints are available in [HuggingFace](). 


## Training

To instruction-tune the base models, run the following:
```
bash train.sh $base_model $data_dir $expt_dir $num_epochs $tasks
```
- `$base_model` specifies the base model (either `mistralai/Mistral-7B-Instruct-v0.3` or `meta-llama/Llama-3.1-8B-Instruct`)
- `$data_dir` specifies the path to json files or HuggingFace dataset hub `NingLab/MoMUInstruct`
- `$expt_dir` specifies the path to the directory where the finetuned checkpoints and lora weights will be saved
- `$num_epochs` specifies the number of epochs
- `$tasks` specifies the list of tasks to be used for instruction-tuning
    - example 1: "['bbbp+drd2+qed']" to train a task-specific model optimizing only this task
    - example 2: "['bbbp','drd2','qed','bbbp+qed','bbbp+drd2','drd2+qed','bbbp+drd2+qed']" to train a generalist model on the superset of 3 properties: BBBP, DRD2 and QED

For faster runs, download the base models once locally in a folder and provide the local path into `$base_model`. Please download the base models: [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) and [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) following the HuggingFace instructions.  

## Inference

For model inference, run the following command:

```
bash inference.sh $base_model $data_dir $lora_weights $output_dir $task $num_seq $setting
```
- `$base_model` specifies the base model (either `mistralai/Mistral-7B-Instruct-v0.3` or `meta-llama/Llama-3.1-8B-Instruct`)
- `$data_dir` specifies the path to json files or HuggingFace dataset hub `NingLab/MoMUInstruct`
- `$lora_weights` specifies the path containing the adapter weights of the instruction-tuned model
- `$output_dir` specifies the output path where the LLM generated responses will be stored in JSON format
- `$num_seq` specifies the number of generated responses for each prompt (we set it to 20 in our experiments)
- `$setting` specifies whether to use seen or unseen instruction
    - Permitted values: `seen` or `unseen` 