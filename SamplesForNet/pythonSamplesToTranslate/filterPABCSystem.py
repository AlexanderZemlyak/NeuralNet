import json

if __name__ == "__main__":
    
    filename = 'C:\PABCWork.NET\SamplesForNet\samplesPABCSystem.json'

    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()

    data = json.loads(json_data)

    

    new_data = []

    for el in data:
