import json
import re
from collections import Counter

if __name__ == '__main__':

    with open(r'C:\Users\alex\Desktop\Дипломная\NeuralNet\SamplesForNet\samplesPABCSystem.json', 'r', encoding='utf-8') as f:
        all_samples = json.loads(f.read())

    # all_samples = all_samples[:len(all_samples) // 2 + 1]

    # Функция для извлечения названия функции/процедуры
    def extract_function_name(output):
        # Ищем function Name(... или procedure Name(...
        match = re.search(r'\b(function|procedure)\s+([a-zA-Z_][a-zA-Z0-9_]*)', output)
        if match:
            return match.group(2)  # возвращаем только имя
        return None
    
    # Собираем все названия функций
    function_names = []
    for sample in all_samples:
        if 'output' in sample:
            name = extract_function_name(sample['output'])
            if name:
                function_names.append(name)
    
    # Считаем повторения
    name_counter = Counter(function_names)
    
    # Выводим функции, которые повторяются более 1 раза
    print("Функции/процедуры, повторяющиеся более 1 раза:")
    print("-" * 50)
    
    for name, count in name_counter.items():
        if count > 1:
            print(f"{name}: {count} раз")
            
            