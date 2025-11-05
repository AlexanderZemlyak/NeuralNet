import json
import os
import glob

def merge_json_files(input_folder, output_file):
    """
    Объединяет все JSON файлы из папки в один файл с объединенным массивом
    
    Args:
        input_folder (str): Путь к папке с JSON файлами
        output_file (str): Путь к выходному файлу
    """
    
    # Проверяем существование папки
    if not os.path.exists(input_folder):
        print(f"Ошибка: Папка '{input_folder}' не существует")
        return
    
    # Находим все JSON файлы в папке
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print("В папке не найдено JSON файлов")
        return
    
    print(f"Найдено {len(json_files)} JSON файлов:")
    
    merged_data = []
    
    # Читаем и объединяем данные из каждого файла
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Проверяем, что данные являются массивом
                if isinstance(data, list):
                    merged_data.extend(data)
                    print(f"  {os.path.basename(file_path)}: добавлено {len(data)} элементов")
                else:
                    print(f"  Предупреждение: {os.path.basename(file_path)} не содержит массив, пропускаем")
                    
        except json.JSONDecodeError as e:
            print(f"  Ошибка чтения {os.path.basename(file_path)}: {e}")
        except Exception as e:
            print(f"  Ошибка обработки {os.path.basename(file_path)}: {e}")
    
    # Сохраняем объединенные данные в новый файл
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(merged_data, file, ensure_ascii=False, indent=2)
        
        print(f"\nУспешно! Объединенный файл создан: {output_file}")
        print(f"Всего элементов в объединенном массиве: {len(merged_data)}")
        
    except Exception as e:
        print(f"Ошибка сохранения файла: {e}")

# Пример использования
if __name__ == "__main__":
    # Укажите путь к папке с JSON файлами
    input_folder = "C:\PABCWork.NET\SamplesForNet\\newSamples"  # Замените на ваш путь
    
    # Укажите имя выходного файла
    output_file = "all_samples.json"
    
    # Вызываем функцию объединения
    merge_json_files(input_folder, output_file)
