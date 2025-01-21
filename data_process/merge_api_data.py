import json
# apify used to get tweets from twitter profiles. The data that is extracted is in json format.
# link 1 : https://apify.com/epctex/twitter-profile-scraper
# link 2 :https://apify.com/danek/twitter-timeline


def create_json_from_multiple_users(file_path):
    file_path = ""

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_data = []

    for tweet in data:
        important_data = {
            "username": tweet["user"]["screen_name"],
            "followers_count": tweet["user"]["followers_count"],
            "bio": tweet["user"]["description"],
            "tweet_text": tweet["full_text"],
            "tweet_date": tweet["created_at"],
            "language": tweet["lang"],
            "favorite_count": tweet["favorite_count"],
            "retweet_count": tweet["retweet_count"],
            "retweeted": tweet["retweeted"],
            "verified": tweet["user"]["verified"]
        }
        extracted_data.append(important_data)

    return extracted_data


def create_json_from_user(file_path):
    # usernames of the users in the data, this is used to identify the user.
    # you can change this to the actual usernames of the users
    usernames = {"p1": "user1", "p2": "user2",
                 "p3": "user3", "p4": "user4", "p5": "user5"}

    extracted_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # loop through the data and extract the important information
        for tweet in data:
            tweet_id = tweet["tweet_id"]
            tweet_text = tweet["text"]
            favorite_count = tweet["favorites"]
            retweet_count = tweet["retweets"]
            username = usernames[file_path.split("\\")[1].split(".")[0]]
            tweet_date = tweet["created_at"]
            language = tweet["lang"]
            extracted_data.append({
                "tweet_id": tweet_id,
                "username": username,
                "tweet_text": tweet_text,
                "tweet_date": tweet_date,
                "favorite_count": favorite_count,
                "retweet_count": retweet_count,
                "language": language
            })

    return extracted_data

# change all occurences of "occupation" to the name of the folder containing the data. -> (real occupation names)


file_path = "occupation"
multiple_path = file_path+"\\p6-10.json"
multiple_user = create_json_from_multiple_users(file_path)

user1 = create_json_from_user(file_path+"\\p1.json")
user2 = create_json_from_user(file_path+"\\p2.json")
user3 = create_json_from_user(file_path+"\\p3.json")
user4 = create_json_from_user(file_path+"\\p4.json")
user5 = create_json_from_user(file_path+"\\p5.json")

# merge the individual user data into one list
users = user1+user2+user3+user4+user5

with open('occupation_users.json', 'w', encoding='utf-8') as file:
    json.dump(users, file, ensure_ascii=False, indent=4)

with open('occupation_multiple_user.json', 'w', encoding='utf-8') as file:
    json.dump(multiple_user, file, ensure_ascii=False, indent=4)
