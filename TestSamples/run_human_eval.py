import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import tempfile
import os
import re
import json

SPECIAL_TOKENS = ["!type!", "!/type!"]

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

def load_model_and_tokenizer(model_path = "deepseek-ai/deepseek-coder-1.3b-instruct", lora_path = None):
    """Загрузка модели и токенизатора"""

    # Загружаем основную модель
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)

    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    add_special_tokens(tokenizer, model)

    # Загружаем LoRA адаптеры
    if lora_path != None:
      model = PeftModel.from_pretrained(model, lora_path)

    # Переводим в режим инференса
    model.eval()

    model = torch.compile(model)

    return model, tokenizer

from peft.peft_model import PeftModelForCausalLM
def build_deepseekcoder_instruction(languge: str, question: str):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())

def generate_one_completion(model, tokenizer, example, max_new_tokens=1024, temperature=0.0, lang = 'python'):
    """Генерация одного решения для задачи"""
    prompt = example['prompt']

    prompt = build_deepseekcoder_instruction(lang, prompt)

    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt }],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"

    # prompt_length = inputs.shape[1]

    # stop_criteria = StopStringCriteria(tokenizer, ["\n# test", "\n# Test", '\n# Example usage:', '\n# Usage:', '\nprint('], prompt_length)
    # PeftModelForCausalLM().generate()
    with torch.no_grad():
        outputs = model.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=stop_id,
            eos_token_id=stop_id
            # stopping_criteria=StoppingCriteriaList([stop_criteria])
        )

    generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=False)

    example['output'] = generated_text

    return example

languge_settings = {
    'python': {
        'full_name': 'Python',
        'indent': 4,
    },
    'pascalabc.net' : {
        'full_name': 'PascalABC.NET',
        'indent': 0
    },
    'pascal' : {
        'full_name': 'Pascal',
        'indent': 0
    }
}

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    elif lang.lower() == 'pascalabc.net' or lang.lower() == 'pascal':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("function ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix

    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def extract_generation_code(example: str, lang_code: str, verbose: bool=False):
    task_id = example['task_id']
    output = example.get('output', example.get("gpt_completion"))
    question = example["prompt"].strip()
    setting = languge_settings[lang_code]
    lang = setting['full_name']
    indent = setting['indent']

    try:
        if output.strip().endswith("<|EOT|>"):
          output = output.replace("<|EOT|>", "")

        if output.find('```') == -1:
          code_block = output
        else:
          code_block: str = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        if verbose:
            print(">>> Task: {}\n{}".format(task_id, code_block))

        # Remove main
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]

        func_name, func_prefix = get_function_name(question, lang)

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent-1] == ' ':
                indent += 1

            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)
        except:
            start = 0
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)

        body = code_block[start:end]

        if lang_code.lower() in ['php', 'ts', 'js']:
            body += '\n' + ' '*indent + '}'

        generation = func_prefix + '\n' + body + '\n'
        example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        example['generation'] = example['prompt'] + '\n' + output

    return example

def run_human_eval_benchmark(lora_path = None, lang = 'python'):
    """Запуск HumanEval benchmark"""
    print("Загрузка модели и токенизатора...")
    model, tokenizer = load_model_and_tokenizer(lora_path=lora_path)

    print("Загрузка задач HumanEval...")
    problems = read_problems()

    print("Генерация решений...")
    samples = []

    for task_id, problem in problems.items():
        print(f"Обработка задачи: {task_id}")

        try:
            problem_with_completion = generate_one_completion(model, tokenizer, problem)
            problem_with_cleaned_completion = extract_generation_code(problem_with_completion, lang)

            samples.append({
                "task_id": task_id,
                "completion": problem_with_cleaned_completion['generation']
            })

            print(f"✅ Завершено: {task_id}")

        except Exception as e:
            print(f"❌ Ошибка в задаче {task_id}: {str(e)}")
            # Добавляем пустое решение в случае ошибки
            samples.append({
                "task_id": task_id,
                "completion": "# Error in generation"
            })

    # Сохраняем результаты во временный файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        write_jsonl(f.name, samples)
        temp_file = f.name

    try:
        # Оцениваем корректность решений
        print("Оценка результатов...")
        results = evaluate_functional_correctness(
            temp_file,
            k=[1],  # оцениваем только первое решение
            n_workers=4,
            timeout=3.0  # время выполнения в секундах
        )

        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ HUMANEVAL BENCHMARK")
        print("="*50)
        print(f"Pass@1: {results['pass@1']:.3f}")

        # Дополнительная информация
        if 'details' in results:
            details = results['details']
            passed = sum(1 for d in details if d['passed'])
            total = len(details)
            print(f"Пройдено задач: {passed}/{total}")
            print(f"Процент успеха: {passed/total*100:.2f}%")

    finally:
        # Удаляем временный файл
        os.unlink(temp_file)

    return results

def test_single_problem(model_path="deepseek-ai/deepseek-coder-1.3b-instruct", lora_path = None, lang='python', standard_problems=True, problems=None, ind=0):
    """Тестирование на одной задаче для отладки"""
    model, tokenizer = load_model_and_tokenizer(model_path=model_path, lora_path=lora_path)

    if standard_problems:
      problems = read_problems()
      # Берем первую задачу для примера
      task_id = list(problems.keys())[ind]
      problem = problems[task_id]
    else:
      problem = problems[ind]


    print(f"Задача {ind + 1}")
    print("Промпт:")
    print(problem["prompt"])
    print("\n" + "="*50)

    problem_with_completion = generate_one_completion(model, tokenizer, problem, 2048, lang=lang)
    problem_with_cleaned_completion = extract_generation_code(problem_with_completion, lang)

    print("Сгенерированный код:")
    print(problem_with_cleaned_completion['generation'])

def test_all_problems(model_path="deepseek-ai/deepseek-coder-1.3b-instruct", lora_path = None, lang='python', standard_problems=True, problems=None):
    """Тестирование на одной задаче для отладки"""
    model, tokenizer = load_model_and_tokenizer(model_path, lora_path)

    if standard_problems:
      problems = read_problems()

    results = []

    for ind in range(len(problems)):
      print(f"Задача {ind + 1}")
      print("Промпт:")
      print(problems[ind]["prompt"])
      print("\n" + "="*50)

      problem_with_completion = generate_one_completion(model, tokenizer, problems[ind], lang=lang)
      problem_with_cleaned_completion = extract_generation_code(problem_with_completion, lang)

      print("Сгенерированный код:")
      print(problem_with_cleaned_completion['generation'])
      results.append(problem_with_cleaned_completion['generation'])

    return results

def load_custom_problems(path):
  with open(path, 'r', encoding='utf-8') as f:
      json_array = json.load(f)

  jsonl_filename = "human_eval.jsonl"
  with open(jsonl_filename, 'w', encoding='utf-8') as f:
      for obj in json_array:
          json_line = json.dumps(obj, ensure_ascii=False)
          f.write(json_line + '\n')

  problems = []

  with open(jsonl_filename, 'r', encoding='utf-8') as f:
      for line in f:
          line = line.strip()
          if line:  # пропускаем пустые строки
              problem = json.loads(line)
              problems.append(problem)

  print(f"Загружено {len(problems)} задач")
  return problems

if __name__ == "__main__":
    problems = load_custom_problems('human_eval_pascal.json')

    test_single_problem(lora_path="../LoRa/checkpoint-100", lang='pascalabc.net', standard_problems=False, problems=problems, ind=0)

    # completions = test_all_problems("deepseek-ai/deepseek-coder-1.3b-instruct", "../LoRa/checkpoint-100", 'pascalabc.net', False, problems)

    filename = 'completions_LoRa21_05_2026_10000samples_normal+enriched_checkpoint-100(16-1).json'
    # with open(filename, 'w', encoding='utf-8') as f:
    #     json.dump(completions, f, ensure_ascii=False, indent=2)

