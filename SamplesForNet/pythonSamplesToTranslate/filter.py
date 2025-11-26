import json
import random

if __name__ == "__main__":
    
    filename = 'C:\PABCWork.NET\SamplesForNet\pythonSamplesToTranslate\educational_instruct_train.json'
  
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()

    data = json.loads(json_data)

    data = random.sample(data, 10000)

    entry_points = dict()

    filtered_data = []

    for i, item in enumerate(data):

        if i == 1800:
            break

        c = entry_points.get(item['entry_point'])
        if c != None:
            if c == 3:
                continue
            else:
                entry_points[item['entry_point']] += 1
                filtered_data.append(item)
        else:
            entry_points[item['entry_point']] = 1
            filtered_data.append(item)


    output_file = 'educational_instruct_filtered.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([{'instuction' : item['instruction'], 'output' : item['code'] } for item in filtered_data], f, indent=2, ensure_ascii=False)
