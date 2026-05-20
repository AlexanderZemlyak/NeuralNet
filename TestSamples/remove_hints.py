import re
import json

with open('completions_LoRa08_05_2026_5000samples_enriched_checkpoint-100_full_prompt(16-4).json', 'r', encoding='utf-8') as f:
        completions_data = json.loads(f.read())

new_completions = []

for item in completions_data:
    item = re.sub('!type![^!]+!/type!', '', item)
    new_completions.append(item)

with open('completions_LoRa08_05_2026_5000samples_enriched_checkpoint-100_full_prompt(16-4)2.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_completions, indent=4))

