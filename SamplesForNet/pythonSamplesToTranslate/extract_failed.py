import json
import os
import re

def generate_pascal_files_from_json(json_data):
    """Генерирует файлы PascalABC.NET из JSON данных"""
    try:
        data = json.loads(json_data)

        for i, item in enumerate(data):
            solution = item['output']         

            file_content = generate_pascal_file_content(solution)
            
            # print(stripped_solution)

            # Сохраняем файл
            filename = str(i) + '.pas'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            if (os.path.exists(str(i) + '.exe')):
                os.remove(str(i) + '.exe')        
        
            print(f'Создан файл: {filename}')
          
    except Exception as e:
        print(f'Ошибка при обработке JSON: {e}')

def generate_pascal_file_content(solution):
    """Генерирует содержимое Pascal файла"""
    content = f'''
// Auto-generated from JSON
// {"{$zerobasedstrings}"}

{solution}

begin
end.

'''
    return content

if __name__ == "__main__":
    
    filename = 'pascal_translated1801-3110.json'

    new_json_data = []

    with open(filename, 'r', encoding='utf-8') as f:
         json_data = f.read()

    data = json.loads(json_data)

    for i in range(1310):
        if not os.path.exists('..\\' + str(i) + '.exe'):
            new_json_data.append(data[i])

    file_content = json.dumps(new_json_data)

    with open('..\\temp\\failed.json', 'w', encoding='utf-8') as f:
        f.write(file_content)
