from __future__ import division

def captializedWordsInTweet(tweet):
    words=["I","A"]
    count=0
    if(len(tweet) != 0):
        for w in tweet.split():
            if w.isupper():
                if w not in words:
                    count =count+1;
    return count

def exclamationCount(tweet):
    tweetWords = tweet.split()
    count=0
    for word in tweetWords:
        if (word.count("!")>2):
            count+=1
    return  count

def questionMarkCount(tweet):
    tweetWords = tweet.split()
    count=0
    for word in tweetWords:
        if (word.count("?") > 2):
            count+=1
    return  count

def capitalCountInAWord(tweet):
    count=0
    if (len(tweet) != 0):
        for c in tweet:
            if (str(c).isupper()):
                count=count+1
    return count


