import json

if __name__ == "__main__":
    
    filename = 'C:\\Users\\alex\\Desktop\\Дипломная\\NeuralNet\\SamplesForNet\\samplesPABCSystem.json'

    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()

    data = json.loads(json_data)

    vars_to_exclude = ['WRITELN_IN_BINARYFILE_ERROR_MESSAGE', 'InternalNullBasedArrayName', 'FILE_NOT_ASSIGNED',
                       'FILE_NOT_OPENED', 'FILE_NOT_OPENED_FOR_READING', 'FILE_NOT_OPENED_FOR_WRITING',
                       '[System.Diagnostics.DebuggerStepThrough]']

    new_data = []

    for el in data:
        found = False
        for var in vars_to_exclude:
            if el['output'].find(var) != -1:
                found = True
        
        if not found:
            new_data.append(el)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data))

