import os,sys
from typing import List
import pandas as pd

import fire
import torch 
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          DataCollatorForSeq2Seq
)
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict
)
from datasets import Dataset, load_dataset
from datasets import disable_caching
from transformers import DataCollatorForSeq2Seq

disable_caching()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prompter import Prompter

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path, weights_only=True)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control
    


def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    tasks: List[str] = [],
    # dev_data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    cutoff_len: int = 4096,
    train_set_size: int = None,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    #warmup_steps: int = 100,
    warmup_ratio: float = 0.1, 
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    #lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj","up_proj", "down_proj", "lm_head"],
    lora_weight_path: str = None,
    load_in_8bit: bool = False,
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    gradient_checkpointing: bool = False,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "mistral",
    opt_type: str = "simple",
    optim: str = "adamw_torch",
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template: {prompt_template_name}\n"
            f"tasks: {tasks}\n"
            f"opt_type: {opt_type}\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"train_set_size: {train_set_size}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"lora_weight_path: {lora_weight_path}\n"
            f"load_in_8bit: {load_in_8bit}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"gradient_checkpointing: {gradient_checkpointing}\n"
            f"optim: {optim}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(opt_type)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and os.environ["WANDB_PROJECT"]
    )
    # only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
        #attn_implementation="flash_attention_2"
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    bf16 = True

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, return_tensors=None, padding=False, truncation=True, max_length=cutoff_len)
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point):
        
        full_prompt = prompter.generate_prompt(
            data_point,
            True,
            'ins',
            opt_type,
            add_response=True,
        )

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point,
                True,
                'ins',
                opt_type,
                add_response=False,
            )
            
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token)
            
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # TODO: Speed up?
        return tokenized_full_prompt
    

    if tasks:
        data = load_dataset("vdey/MuMOInstruct")['train']
        data = pd.DataFrame(data)

        train_data = data[(data['split'] == 'train') & (data['task'].isin(tasks))]
        val_data = data[(data['split'] == 'val') & (data['task'].isin(tasks))]
        train_data = train_data.groupby('task').head(train_set_size).reset_index(drop=True)
        val_data = val_data.groupby('task').head(val_set_size).reset_index(drop=True)

        # for each task, print a demo prompt        
        for task in tasks:
            print(f"Task: {task}")
            example_data = train_data[train_data['task'] == task].iloc[0]
            print(prompter.generate_prompt(
                example_data,
                True,
                'ins',
                opt_type,
                add_response=True,
            ))
            
        train_data = Dataset.from_pandas(train_data)
        val_data = Dataset.from_pandas(val_data)

    else:
        sys.exit("Please specify a --tasks")


    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    
    if lora_weight_path:
        model.load_adapter(lora_weight_path, adapter_name="default")
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model.load_adapter(resume_from_checkpoint, adapter_name="default")

    model.print_trainable_parameters()

    train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
    val_data = val_data.map(generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    #count the number of tokens
    num_tokens = sum(len(data['input_ids']) for data in train_data)
    print(f'#tokens: {num_tokens / 1000.0:.1f}k')
    
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            gradient_checkpointing=gradient_checkpointing,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16, 
            logging_steps=1,
            optim=optim,
            eval_strategy="epoch" if val_set_size > 0 else "no",
            save_strategy="steps", #"epoch",
            save_steps=1000,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorForSeq2Seq(
           tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    model.save_pretrained(output_dir)
    pytorch_model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save({}, pytorch_model_path)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)