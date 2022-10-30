# Creating the filtered data list
import json

file_path = "./movies.txt"
movieObjectCount = 10
dataset = "movie_reviews.csv" 

file = open(file_path, encoding="utf8")
file.close()

file = open(file_path, encoding="utf8")
movie_list = []

def filterData(lineText):
    return ":".join(lineText.split(":")[1:])

for i in range(movieObjectCount):
    movieObjectData = {
    "productId": "",
    "userId": "",
    "profileName": "",
    "helpfulness": "",
    "score": "",
    "time": "",
    "summary": "",
    "text": ""
}
    movieObjectData["productId"] = filterData(file.readline())
    movieObjectData["userId"] = filterData(file.readline())
    movieObjectData["profileName"] = filterData(file.readline())
    movieObjectData["helpfulness"] = filterData(file.readline())
    movieObjectData["score"] = filterData(file.readline())
    movieObjectData["time"] = filterData(file.readline())
    movieObjectData["summary"] = filterData(file.readline())
    movieObjectData["text"] = filterData(file.readline())
    file.readline()
    movie_list.append(movieObjectData)

file.close()


movie_summary_text_list = []
for movie in movie_list:
    movie_summary_text_list.append({
        "summary": movie["summary"],
        "text": movie["text"]
    })

# for movie in movie_summary_text_list:
#     print(json.dumps(movie, indent=2))


# Converting JSON list to CSV
import pandas as pd

df = pd.DataFrame(movie_summary_text_list)
df.to_csv(dataset, index=False)


# Reading the CSV file
import pandas as pd

df = pd.read_csv(dataset)
df.head()

# finding for the amount of missing data in the dataset
df.isnull().sum()


