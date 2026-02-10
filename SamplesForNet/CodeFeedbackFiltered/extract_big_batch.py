import json

if __name__ == "__main__":

    with open('CodeFeedback_Sharp.json', 'r', encoding='utf-8') as f:
        data = f.read()

        json_data = json.loads(data)

    # Переменный номер!
    batch_number = 2

    batch_size = 250

    start_index = (batch_number-1)*batch_size
    end_index = start_index + batch_size

    with open(f'CodeFeedback_Sharp_batch{batch_number}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps([{ 'instruction' : sample['query'], 'output' : sample['answer'] }
                            for sample in json_data[start_index:end_index]], indent=4))
    