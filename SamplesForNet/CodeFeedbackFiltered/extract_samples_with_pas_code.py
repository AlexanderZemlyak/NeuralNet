import json

if __name__ == "__main__":

    with open('CodeFeedback_translated_batch2.json', "r", encoding="utf-8") as f:
        samples = json.loads(f.read())

    new_samples = []

    for sample in samples:
        
        solution = sample['output']

        start_index = solution.find("```pascal")
        offset = 9

        solution_parts = []

        while start_index != -1:
            solution_parts.append(solution[start_index+offset:solution.index("```", start_index+offset)])
            start_index = solution.find("```pascal", start_index + 1)

        if len(solution_parts) > 0:
            solution = str.join('\n', solution_parts)

        new_samples.append({ 'instruction': sample['instruction'], 'output': solution })

    
    with open('CodeFeedback_translated_batch2_only_code.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_samples))
