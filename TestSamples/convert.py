import json
import re

def snake_to_pascal_case(snake_str):
    """Convert snake_case to PascalCase"""
    return ''.join(word.capitalize() for word in snake_str.split('_'))

def convert_function_declaration(prompt, test):
    """Convert ALL function names and parameters from snake_case to PascalCase in the prompt"""
    
    # Регулярное выражение для поиска функции и параметров
    pattern = r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*:\s*(.+?);'
    
    # Находим ВСЕ совпадения в prompt
    matches = list(re.finditer(pattern, prompt))
    
    if not matches:
        return prompt  # Если не нашли функции, возвращаем как есть
    
    converted_prompt = prompt
    
    # Обрабатываем каждую найденную функцию
    for match in matches:
        function_name = match.group(1)
        parameters = match.group(2)
        return_type = match.group(3)
        
        # Конвертируем имя функции
        pascal_function_name = snake_to_pascal_case(function_name)
        
        # Конвертируем параметры
        def convert_parameter(param):
            param = param.strip()
            # Разделяем имя параметра и тип
            if ':' in param:
                name_part, type_part = param.split(':', 1)
                # Конвертируем только имя параметра (до двоеточия)
                pascal_name = snake_to_pascal_case(name_part.strip())
                return f'{pascal_name[0].lower() + pascal_name[1:]} : {type_part.strip()}'
            return param
        
        # Обрабатываем все параметры
        if parameters:
            param_list = [param.strip() for param in parameters.split(';')]
            converted_params = '; '.join(convert_parameter(param) for param in param_list)
        else:
            converted_params = ''
        
        # Собираем новую сигнатуру
        new_signature = f'function {pascal_function_name}({converted_params}): {return_type};'
        
        # Заменяем в prompt
        converted_prompt = converted_prompt.replace(match.group(0), new_signature)
        
        # Также заменяем вызовы этой функции в примерах
        converted_prompt = converted_prompt.replace(f'>>> {function_name}(', f'>>> {pascal_function_name}(')
        test = test.replace(f'{function_name}(', f'{pascal_function_name}(')
    
    return converted_prompt, test

def process_json_file(input_file, output_file=None):
    """Process JSON file and convert function names to PascalCase"""
    
    # Читаем JSON файл
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Обрабатываем каждый элемент в массиве
    for item in data:
        if 'prompt' in item:
            item['prompt'], item['test'] = convert_function_declaration(item['prompt'], item['test'])
        
        # Также обновляем entry_point если нужно
        if 'entry_point' in item:
            item['entry_point'] = snake_to_pascal_case(item['entry_point'])
    
    # Сохраняем результат
    if output_file is None:
        output_file = input_file.replace('.json', '_pascal.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Обработанный файл сохранен как: {output_file}")
    return data

# Пример использования
if __name__ == "__main__":
    input_filename = "human_eval_pascal_old.json"  # Замените на путь к вашему файлу
    output_filename = "human_eval_pascal.json"
    process_json_file(input_filename, output_filename)