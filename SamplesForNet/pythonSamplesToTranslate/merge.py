import json

if __name__ == "__main__":
    
    filename = 'C:\\PABCWork.NET\\SamplesForNet\\temp\\failed.json'

    new_json_data = []

    with open(filename, 'r', encoding='utf-8') as f:
         json_data = f.read()

    data = json.loads(json_data)

    filename2 = 'C:\\PABCWork.NET\\SamplesForNet\\pythonSamplesToTranslate\\pascal_translated_batch7.json'

    with open(filename2, 'r', encoding='utf-8') as f:
         json_data2 = f.read()

    data2 = json.loads(json_data2)

    for item in data:
        s = item['id']
        for i, item2 in enumerate(data2):
            s2 = item2['id']
            if s == s2:
                data2[i]['output'] = item['output']
                break

    file_content = json.dumps(data2)

    with open(filename2, 'w', encoding='utf-8') as f:
        f.write(file_content)