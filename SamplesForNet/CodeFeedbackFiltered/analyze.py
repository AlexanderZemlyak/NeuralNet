import json

if __name__ == "__main__":

    with open('CodeFeedback_Sharp.json', 'r', encoding='utf-8') as f:
        data = f.read()

        json_data = json.loads(data)

    new_data = []

    for i in range(250):
        item = json_data[i]
        new_data.append({'instruction' : item['query'], 'output' : item['answer']})

    with open('CodeFeedback_Sharp_batch1.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))
