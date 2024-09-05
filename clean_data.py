import re

# Step 1: Define a function to clean the file
def clean_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # Step 2: Replace invalid control characters
    # This pattern targets control characters that are not allowed in JSON (except \n, \r, and \t)
    cleaned_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)

    # Step 3: Escape invalid backslashes
    # This pattern finds backslashes that are not followed by a valid escape character
    cleaned_content = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', cleaned_content)

    # Step 4: Write the cleaned content to a new file
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write(cleaned_content)

# Step 5: Clean both JSON files
clean_json_file('data/records_dbp15k_en.json', 'data/records_dbp15k_en_clean.json')
clean_json_file('data/records_dbp15k_fr.json', 'data/records_dbp15k_fr_clean.json')
