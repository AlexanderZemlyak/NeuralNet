import json
import os

if __name__ == "__main__":

    batch_number = 2

    with open(f'CodeFeedback_Sharp_batch{batch_number}.json', 'r', encoding='utf-8') as f:
        data = f.read()

        json_data = json.loads(data)

    c = 1

    os.mkdir(f'CodeFeedback_Sharp_batch{batch_number}_1-{batch_number}_13')

    for i in range(0, 250, 20):
        with open(f'CodeFeedback_Sharp_batch{batch_number}_1-{batch_number}_13\\CodeFeedback_Sharp_batch{batch_number}_{c}.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(json_data[i:i+20], indent=4))
        c += 1
