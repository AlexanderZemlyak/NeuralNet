import json

if __name__ == '__main__':

    filename1 = 'C:\\PABCWork.NET\\SamplesForNet\\pythonSamplesToTranslate\\test.json'

    filename2 = 'C:\\PABCWork.NET\\SamplesForNet\\pythonSamplesToTranslate\\pascal_translated_batch1.json'

    with open(filename1, 'r', encoding='utf-8') as f:
        data1 = json.loads(f.read())

    with open(filename2, 'r', encoding='utf-8') as f:
        data2 = json.loads(f.read())

    new_json_data = []

    for item in data1:
        if len(list(filter(lambda i: i['instruction'] == item['instruction'] or i['output'] == item['output'], data1))) > 1:
            print(item['instruction'])
            print()

    for item in data1:
        if next((el for el in data2 if el['instruction'][-20:] == item['instruction'][-20:]), None) == None:
            new_json_data.append(item)

    file_content = json.dumps(new_json_data)

    with open('C:\\PABCWork.NET\\SamplesForNet\\temp\\missed.json', 'w', encoding='utf-8') as f:
        f.write(file_content)
    


