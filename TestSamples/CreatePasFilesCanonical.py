import json
import os

def generate_pascal_files_from_json(json_data):
    """Генерирует файлы PascalABC.NET из JSON данных"""
    try:
        data = json.loads(json_data)
        
        for item in data:
            task_id = item['task_id']
            prompt = item['prompt']
            entry_point = item['entry_point']
            canonical_solution = item['canonical_solution'].replace('"', '\'')            
            test = item['test'].replace('"', '\'')
            
            # Создаем содержимое файла
            file_content = generate_pascal_file_content(task_id, prompt, entry_point, canonical_solution, test)
            
            # Сохраняем файл
            filename = task_id.replace('/', '_') + '.pas'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(file_content)
                
            if (os.path.exists(task_id.replace('/', '_') + '.exe')):
                os.remove(task_id.replace('/', '_') + '.exe')      
            
            print(f'Создан файл: {filename}')
          
    except Exception as e:
        print(f'Ошибка при обработке JSON: {e}')

def generate_pascal_file_content(task_id, prompt, entry_point, canonical_solution, test):
    """Генерирует содержимое Pascal файла"""
    content = f'''// {task_id}
// Auto-generated from JSON

procedure _Assert(cond: boolean);
begin
  if not cond then
    raise new Exception();
end;

{prompt}
{canonical_solution}

{test.replace('assert(', '  assert(')}
'''
    return content

# Пример использования
if __name__ == "__main__":
    filename = 'C:\PABCWork.NET\TestSamples\human_eval_pascal.json'
  
    with open(filename, 'r', encoding='utf-8') as f:
            json_data = f.read()
    
    generate_pascal_files_from_json(json_data)
    print('Генерация файлов завершена!')
