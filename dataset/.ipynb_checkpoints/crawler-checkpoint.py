import requests
from bs4 import BeautifulSoup
import re
import json
import argparse

def getMostReplayed(youtube_id):
    url = f'https://www.youtube.com/watch?v={youtube_id}'
    response = requests.get(url).text
    soup = BeautifulSoup(response, 'html.parser')
    data = re.search(r'var ytInitialData = ({.*?});', soup.prettify()).group(1)
    data = json.loads(data)
    #print(data['frameworkUpdates']['entityBatchUpdate']['mutations'][0]['payload']['macroMarkersListEntity']['markersList']['markers'])
    # Some video's Most Replayed features disappeared. Therefore, I will use the try-except block 
    #try:
    markersMap = data['frameworkUpdates']['entityBatchUpdate']['mutations'][0]['payload']['macroMarkersListEntity']['markersList']['markers']
    mostReplayed = markersMap if markersMap is not None else None
    return mostReplayed
    #except:
        #print("Most Replayed features are removed or the video is no longer available")
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", help='youtube video id')
    args = parser.parse_args()

    id = args.vid
    most_replayed_statistics = getMostReplayed(id)
    with open(f"dataset/most_replayed_{id}.json", "w") as outfile:
        json.dump(most_replayed_statistics, outfile)

