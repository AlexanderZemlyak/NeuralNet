import json

if __name__ == "__main__":
    
    filename = 'C:\\PABCWork.NET\\SamplesForNet\\pythonSamplesToTranslate\\educational_instruct_filtered2.json'

    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()

    data = json.loads(json_data)

    for i, item in enumerate(data):
        data[i]['id'] = str(i)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))
    
