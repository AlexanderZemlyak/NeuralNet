import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import gc
import transformers
from transformers import Trainer
from datasets import load_dataset

from peft import LoraConfig, get_peft_model, TaskType

import os


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

SPECIAL_TOKENS = ["!type!", "!/type!"]

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,           # Alpha = 2x rank
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # Все attention и MLP слои
    ],
    bias="all",
    modules_to_save=["embed_tokens", "lm_head"]  # Важно! Сохраняем embedding и head
)

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
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-1.3b-instruct")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,  # Уменьшил для Colab
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to deepspeed config"})

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

def add_special_tokens(tokenizer, model):
    """Добавляет специальные токены в токенизатор и расширяет embedding слой модели."""
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': SPECIAL_TOKENS
    })
    
    if num_added > 0:
        print(f"✅ Добавлено специальных токенов: {num_added}")
        print(f"   Токены: {SPECIAL_TOKENS}")
        
        model.resize_token_embeddings(len(tokenizer))
        
        for token in SPECIAL_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"   '{token}' -> id={token_id}")
        
        return True
    else:
        print("⚠️ Токены уже существуют или не добавлены")
        return False

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.remove_unused_columns = False  # Критически важно!

    if training_args.output_dir:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    print('='*100)
    print("Training Arguments:", training_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.empty_cache()
    gc.collect()

    # Загружаем модель с отключенным кэшем
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        use_cache=False  # Отключаем кэш для обучения
    )

    add_special_tokens(tokenizer, model)

    model = get_peft_model(model, lora_config)
    
    # Проверяем, что все параметры на одном устройстве
    devices = {str(param.device) for _, param in model.named_parameters()}
    print(f"Parameters devices: {devices}")

    # Загрузка датасета
    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
    
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer}
    )

    print("Training dataset samples:", len(train_dataset))

    # Демонстрация работы токенизатора с новыми токенами
    test_text = "!type!integer!/type! var x := 42"
    tokens = tokenizer.encode(test_text)
    print(f"\n✅ Тест токенизации: '{test_text}'")
    print(f"   Токены (первые 10): {tokens[:10]}")
    print(f"   Декодировано: {tokenizer.decode(tokens)}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    print("Training completed successfully!")


if __name__ == "__main__":
    import os
    train()