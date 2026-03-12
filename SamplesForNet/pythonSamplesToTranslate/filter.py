import json
import random

if __name__ == "__main__":
    
    filename = 'educational_instruct_train.json'
  
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()

    data = json.loads(json_data)

    data = random.sample(data, 10000)

    filename2 = 'educational_instruct_filtered.json'

    with open(filename2, 'r', encoding='utf-8') as f:
        saved_data = json.loads(f.read())

    entry_points = dict()

    filtered_data = []

    count = 0
    for item in data:

        if count == 1310:
            break

        if item['instruction'] in map(lambda item2: item2['instruction'], saved_data):
            continue

        c = entry_points.get(item['entry_point'])
        if c != None:
            continue
        else:
            entry_points[item['entry_point']] = 1
            filtered_data.append(item)
            count += 1


    output_file = 'educational_instruct_filtered1801-3110.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([{'instruction' : item['instruction'], 'output' : item['code'], 'id': i } for i, item in enumerate(filtered_data)], f, indent=2, ensure_ascii=False)
