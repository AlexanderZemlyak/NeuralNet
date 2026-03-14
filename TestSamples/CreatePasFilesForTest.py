import json
import os
import re

def generate_pascal_files_from_json(json_data, completions_json):
    """Генерирует файлы PascalABC.NET из JSON данных"""
    try:
        data = json.loads(json_data)
        
        completions = json.loads(completions_json)

        for i, item in enumerate(data):
            task_id = item['task_id']
            prompt = item['prompt']
            entry_point = item['entry_point']
            solution = completions[i]            
            test = item['test'].replace('"', '\'')
            
            stripped_solution = solution.strip('\n ')

            stripped_solution = re.sub(r"begin(?:(?!\bbegin\b)[\s\S])+?end\.", '', stripped_solution)

            if stripped_solution.startswith('function') and stripped_solution.find(f'function {entry_point}') != -1:
                 file_content = generate_pascal_file_content(task_id, '', stripped_solution, test)
            else:
                 file_content = generate_pascal_file_content(task_id, prompt, stripped_solution, test)
            
            # print(stripped_solution)

            # Сохраняем файл
            filename = task_id.replace('/', '_') + '.pas'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            if (os.path.exists(task_id.replace('/', '_') + '.exe')):
                os.remove(task_id.replace('/', '_') + '.exe')        
        
            print(f'Создан файл: {filename}')
          
    except Exception as e:
        print(f'Ошибка при обработке JSON: {e}')

def generate_pascal_file_content(task_id, prompt, solution, test):
    """Генерирует содержимое Pascal файла"""
    content = f'''// {task_id}
// Auto-generated from JSON
{"{$zerobasedstrings}"}
uses System;

// type Variant = object;

procedure _Assert(cond: boolean; message: string := nil);
begin
  if not cond then
    Println('Error!!!');
end;

{prompt}
{solution}

{test}
'''
    return content

if __name__ == "__main__":
    
    filename = 'human_eval_pascal.json'
  
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()
    
    with open('completions_LoRa12_03_2026_5000samples_checkpoint-75(16-4).json', 'r', encoding='utf-8') as f:
        completions_data = f.read()

    generate_pascal_files_from_json(json_data, completions_data)
    print('Генерация файлов завершена!')
