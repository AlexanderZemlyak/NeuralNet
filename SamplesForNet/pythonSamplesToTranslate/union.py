import json

if __name__ == "__main__":

    baseFilename = 'C:\\PABCWork.NET\\SamplesForNet\\pythonSamplesToTranslate\\pascal_translated_batch'

    all_samples = []

    for i in range(1, 9):
        with open(baseFilename + str(i) + '.json', 'r', encoding='utf-8') as f:
            all_samples.extend(json.loads(f.read()))

    for i, s in enumerate(all_samples):
        all_samples[i]['id'] = str(i)

    with open(baseFilename[:-6] + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_samples))