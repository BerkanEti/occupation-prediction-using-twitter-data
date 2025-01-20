import json
import os

# Function to load JSON data safely
def load_json(file_path):
    if os.path.exists(file_path):  # Check if file exists
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)  # Load the JSON data
    else:
        print(f"Warning: {file_path} does not exist. Using empty list.")
        return []  # Return an empty list if file is missing

jobs = ["avukat","diyetisyen","doktor","ekonomist","ogretmen","psikolog","sporyorumcusu","tarihci","yazilimci","ziraatmuhendisi"]
for job in jobs :
    print(job)
    data_a_1 = load_json('/job-prediction-from-twitter-data/api-data/' + job + '_multiple_user.json')
    data_a_2 = load_json('/job-prediction-from-twitter-data/api-data/' + job + '_users.json')
    data_b = load_json('/job-prediction-from-twitter-data/mayda-data/' + job + '/' + job + '_mayda.json')
    
    print("size : data_a_1 {}", len(data_a_1))
    print("size : data_a_2 {}", len(data_a_2))
    print("size : data_b {}", len(data_b))

    # Transform a.json data to match the format for b.json
    transformed_data = []
    for entry in data_a_1:
        transformed_data.append({
            "tweet": entry.get("tweet_text", ""),  # Get tweet_text from a.json
            "source": "api"  # Add source as "api"
        })

    for entry in data_a_2:
        transformed_data.append({
            "tweet": entry.get("tweet_text", ""),  # Get tweet_text from a.json
            "source": "api"  # Add source as "api"
        })
    
    print("size : transformed_data {}", len(transformed_data))

    # Merge the data from b.json and transformed a.json data
    merged_data = data_b + transformed_data

    print("size : merged data {}", len(merged_data))

    # Save the merged data into c.json

    with open('/job-prediction-from-twitter-data/merged-data/' + job + '_merged.json', 'w', encoding='utf-8') as file_c:
        json.dump(merged_data, file_c, indent=4, ensure_ascii=False)

    print("Merged data has been written to" + job + "_merged.json")
