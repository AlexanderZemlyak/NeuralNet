import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import gc
import torch.distributed
import transformers
from transformers import Trainer
from datasets import load_dataset

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    return '''
You are a PascalABC.NET coding assistant. Follow these rules:
1. Write only PascalABC.NET code
2. Provide complete programs when possible
3. Use modern PascalABC.NET features
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-1.3b-instruct")  # Изменил на 1.3B для Colab

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,  # Уменьшил для Colab
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    deepspeed: str = field(default=None, metadata={"help": "Path to deepspeed config"})  # Добавил deepspeed аргумент

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def freeze_strategy(model):
    model.model.embed_tokens.requires_grad_(False)      # Эмбеддинги
    model.model.norm.requires_grad_(False)              # Финальная норма
    model.model.rotary_emb.requires_grad_(False)        # Позиционные encoding
    
    for i, layer in enumerate(model.model.layers):
        if i < 18:  # Заморозить первые 18 из 24 слоев (75%)
            # Внимание
            layer.self_attn.q_proj.requires_grad_(False)
            layer.self_attn.k_proj.requires_grad_(False) 
            layer.self_attn.v_proj.requires_grad_(False)
            layer.self_attn.o_proj.requires_grad_(False)
            
            # MLP
            layer.mlp.gate_proj.requires_grad_(False)
            layer.mlp.up_proj.requires_grad_(False)
            layer.mlp.down_proj.requires_grad_(False)
            
            # LayerNorm
            layer.input_layernorm.requires_grad_(False)
            layer.post_attention_layernorm.requires_grad_(False)
    
    return model

def quick_freeze_check(model):
    """Быстрая проверка по группам слоев"""
    
    print("=== БЫСТРАЯ ПРОВЕРКА ===\n")
    
    # Проверка по диапазонам слоев
    layer_ranges = [
        (0, 5, "Ранние слои (0-5)"),
        (6, 11, "Средние слои (6-11)"), 
        (12, 17, "Поздние замороженные (12-17)"),
        (18, 23, "Размороженные (18-23)")
    ]
    
    for start, end, desc in layer_ranges:
        print(f"\n{desc}:")
        for i in range(start, end + 1):
            layer = model.model.layers[i]
            # Проверяем один параметр для примера
            q_proj_frozen = not list(layer.self_attn.q_proj.parameters())[0].requires_grad
            mlp_frozen = not list(layer.mlp.gate_proj.parameters())[0].requires_grad
            
            status = "❄️" if q_proj_frozen and mlp_frozen else "🔥"
            print(f"  Слой {i}: {status}")

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Для Colab проверяем доступность GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in Colab!")
    
    print('='*100)
    print("Training Arguments:", training_args)
    print("Model Arguments:", model_args)
    print("Data Arguments:", data_args)
    
    # Инициализация distributed для DeepSpeed
    training_args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    # Добавляем pad token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    torch.cuda.empty_cache()
    gc.collect()

    # Загрузка модели с поддержкой bf16 для DeepSpeed
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,  # Используем bf16 для DeepSpeed
        trust_remote_code=True,
        # device_map="auto"  # Автоматическое распределение по GPU
    )

    model = freeze_strategy(model)

    print("Load model from {} over.".format(model_args.model_name_or_path))

    # Загрузка датасета
    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
        
    # Токенизация датасета - УМЕНЬШАЕМ параметры для Colab
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=1000,  # Уменьшил с 3000
        num_proc=4,       # Уменьшил с 32 для Colab
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    print("Training dataset samples:", len(train_dataset))
    for index in random.sample(range(len(train_dataset)), 2):  # Уменьшил с 3
        print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
        print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    # Добавляем gradient checkpointing для экономии памяти
    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module
    )

    quick_freeze_check(model)

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    print("Training completed successfully!")


if __name__ == "__main__":
    import os
    train()