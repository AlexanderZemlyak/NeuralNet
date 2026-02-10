import json

def jsonl_to_json_array(input_file, output_file):
    """
    Преобразует JSONL файл в JSON массив
    """
    data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Пропускаем пустые строки
                    data.append(json.loads(line))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Успешно преобразовано! Записано {len(data)} объектов в {output_file}")
        
    except FileNotFoundError:
        print(f"✗ Файл {input_file} не найден")
    except json.JSONDecodeError as e:
        print(f"✗ Ошибка в формате JSON: {e}")
    except Exception as e:
        print(f"✗ Произошла ошибка: {e}")

# Пример использования
if __name__ == "__main__":
    input_file = "CodeFeedback-Filtered-Instruction.jsonl"   # ваш исходный файл
    output_file = "CodeFeedback-Filtered-Instruction.json"  # целевой файл
    jsonl_to_json_array(input_file, output_file)