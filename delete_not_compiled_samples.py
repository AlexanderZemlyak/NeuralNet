import json

if __name__ == "__main__":

    with open('train_data_enriched2.json', 'r', encoding="utf-8") as f:
        samples = json.loads(f.read())

    filtered = []

    for sample in samples:
        if not sample.get('compilationFailed'):
            filtered.append(sample)

    with open('train_data_enriched2_filtered.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(filtered, indent=2, ensure_ascii=False))

    print(f'{len(filtered)} samples saved')