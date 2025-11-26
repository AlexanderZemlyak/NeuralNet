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

            baseDir = 'C:\\PABCWork.NET\\SamplesForNet\\'

            # Сохраняем файл
            filename = baseDir + str(i) + '.pas'
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
    
    filename = 'C:\\PABCWork.NET\\SamplesForNet\\pythonSamplesToTranslate\\pascal_translated.json'
  
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()

    generate_pascal_files_from_json(json_data)
    print('Генерация файлов завершена!')
