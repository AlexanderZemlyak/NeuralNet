import json
import os
import re

def generate_pascal_files_from_json(json_data):
    """Генерирует файлы PascalABC.NET из JSON данных"""
    try:
        data = json.loads(json_data)

        for i, item in enumerate(data):
            solution = item['output']         

            start_index = solution.find("```pascal")
            offset = 9
            # if solution[start_index+9:start_index+16] == "abc.net":
            #     offset += 7

            solution_parts = []

            while start_index != -1:
                solution_parts.append(solution[start_index+offset:solution.index("```", start_index+offset)])
                start_index = solution.find("```pascal", start_index + 1)

            if len(solution_parts) > 0:
                solution = str.join('\n', solution_parts)
            
            file_content = generate_pascal_file_content(solution)

            baseDir = 'D:\\DesktopFiles\\Дипломная\\NeuralNetForPascal\\SamplesForNet\\'

            # Сохраняем файл
            filename = baseDir + str(i) + '.pas'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            if (os.path.exists(baseDir + str(i) + '.exe')):
                os.remove(baseDir + str(i) + '.exe')        
        
            print(f'Создан файл: {filename}')
          
    except Exception as e:
        print(f'Ошибка при обработке JSON: {e}')

def generate_pascal_file_content(solution):

    usesSystem = "\nuses System;"

    if re.match(r"uses (\w| |,|\.)*\bSystem\b(,|;)", solution) != None:
        usesSystem = ""
    
    postfix = 'begin\nend.\n'

    if solution.strip().startswith("##\n") or solution.strip().endswith("end."):
        postfix = ""

    """Генерирует содержимое Pascal файла"""
    content = f'''
// Auto-generated from JSON
{solution}

{postfix}
'''
    return content

if __name__ == "__main__":

    filename = 'all_samples.json'
  
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()

    generate_pascal_files_from_json(json_data)
    print('Генерация файлов завершена!')
