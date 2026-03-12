import json

if __name__ == "__main__":

    filename = 'pascal_translated1801-3110.json'

    with open(filename, 'r', encoding='utf-8') as f:
        samples = json.loads(f.read())

    filename2 = '..\\all_samples.json'

    with open(filename2, 'r', encoding='utf-8') as f:
        all_samples = json.loads(f.read())

    for item in samples:
        all_samples.append({ 'instruction' : item['instruction'], 'output' : item['output'] })

    with open(filename2, 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_samples))