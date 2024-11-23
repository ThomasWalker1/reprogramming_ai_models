import csv

def extract_answer(options, correct):
    options_dict = {}
    for option in options.split(','):
        key, value = option.split(')', 1)
        options_dict[key.strip()] = value.strip()
    return options_dict.get(correct.strip(), '')

input_file = 'math_qa.csv'
output_file = 'math_qa_with_answers.csv'

with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['answer']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in reader:
        try:
            row['answer'] = extract_answer(row['options'], row['correct'])
            writer.writerow(row)
        except Exception as e:
            print(f"Error processing row: {row}. Error: {e}")
            continue