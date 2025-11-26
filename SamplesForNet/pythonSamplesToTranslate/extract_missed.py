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
    
    filename = 'C:\\PABCWork.NET\\SamplesForNet\\pythonSamplesToTranslate\\educational_instruct_filtered2.json'
  
    filename2 = 'C:\\PABCWork.NET\\SamplesForNet\\pythonSamplesToTranslate\\pascal_translated_batch5.json'

    new_json_data = []

    with open(filename, 'r', encoding='utf-8') as f:
         json_data = f.read()

    data1 = json.loads(json_data)

    with open(filename2, 'r', encoding='utf-8') as f:
         json_data = f.read()

    data2 = json.loads(json_data)

    # print(len(set(map(lambda item: item['instruction'], data1))))

    for item in data1:
        if next((el for el in data2 if el['id'] == item['id']), None) == None:
            new_json_data.append(item)

    file_content = json.dumps(new_json_data)

    with open('C:\\PABCWork.NET\\SamplesForNet\\temp\\missed.json', 'w', encoding='utf-8') as f:
        f.write(file_content)
