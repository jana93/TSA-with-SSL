from __future__ import division

def createEmoticonDictionary(filename):
    emo_scores = {'Positive': 0.5, 'Extremely-Positive': 1.0, 'Negative':-0.5,'Extremely-Negative': -1.0,'Neutral': 0.0}
    emo_score_list={}
    fi = open(filename,"r")
    l=fi.readline()
    while l:
        l=l.replace("\xc2\xa0"," ")
        li=l.split(" ")
        l2=li[:-1]
        l2.append(li[len(li)-1].split("\t")[0])
        sentiment=li[len(li)-1].split("\t")[1][:-2]
        score=emo_scores[sentiment]
        l2.append(score)
        for i in range(0,len(l2)-1):
            emo_score_list[l2[i]]=l2[len(l2)-1]
        l=fi.readline()
    return emo_score_list

def emoticonScore(tweet,d):
    s=0.0;
    l=tweet.split(" ")
    nbr=0;
    for i in range(0,len(l)):
        if l[i] in d.keys():
            nbr=nbr+1
            s=s+d[l[i]]
    if (nbr!=0):
        s=s/nbr
    return s

def hastagDict(filename):
    hasScore = {'positive':1,'negative':-1}
    hasScoreList = {}
    fi = open(filename, "r")
    l = fi.readline()
    while l:
        try:
            li = l.split("\t")
            senti = li[1]
            score=hasScore[senti[:-1]]
            hasScoreList[li[0]] = score
        except IndexError:
            print "Error"
        l = fi.readline()
    return hasScoreList

def hashtagWords(tweet,dict):
    l = tweet.split()
    result = []
    for w in l:
        if w[0] == '#':
            for word in dict.keys:
                if (word in w):
                    score = score + word.key
            result.append(w,score)
    return result

## need to implement sentiment hastag
## sentiment lexicons

emo_filename="../resource/emoticon.txt"
has_filename="../resource/sentiHashtag.txt"
tweet="Hi there i am PLAYING Clash of Clans !! ??? :("
emo_dic=createEmoticonDictionary(emo_filename)
has_dic=hastagDict(has_filename)
print has_dic
print emoticonScore(tweet,emo_dic)