import json

def process_tweets_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().split("\n")
    extracted_data = []
    
    for tweet in data:
        important_data = {
            "tweet": tweet,
            "source": "mayda"
        }
        extracted_data.append(important_data)
    
    return extracted_data


jobs = ["avukat","diyetisyen","doktor","ekonomist","ogretmen","psikolog","sporyorumcusu","tarihci","yazilimci","ziraatmuhendisi"]
for job in jobs :
    print(job)
    all_tweets = []
    for i in range(1,6):
        file_path = "/job-prediction-from-twitter-data/mayda-data/" + job  + "/" + str(i) + ".txt"
        all_tweets = all_tweets + process_tweets_from_file(file_path)
        print(len(all_tweets))

    with open("/job-prediction-from-twitter-data/mayda-data/" + job + "/" + job + "_mayda.json", 'w', encoding='utf-8') as file:
        json.dump(all_tweets, file, indent=4, ensure_ascii=False)