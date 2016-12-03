import json

def readFile(fileName):
    reviews = []
    with open(fileName) as f:
        i = 0
        for line in f:
            if i >= 100000:
                break
            a = json.loads(line)
            reviews.append((a['text'],a['stars']))
            i += 1
    return reviews