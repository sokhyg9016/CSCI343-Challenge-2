from os import listdir
from os.path import dirname, join
from collections import Counter
from pandas import DataFrame, read_csv
from re import findall
from numpy import log10, sum
import matplotlib.pyplot as plt

# Get sentiment lexicon
try:
  lexicon = read_csv("data/csv/sentiment_lex.csv")
except:
  print("Error!")

lexicon.columns = ["word", "score"]

# Get series name
series = input("Enter the series name: ").lower()
series_name = series + chr(ord('g') * (1 - (ord(series) & 1))).strip('\x00')

# Get file name
filenames = [f for f in listdir("data/series") if series_name == f.split("1")[0]]

data = ''
for fname in filenames:
  with open("data/series/" + fname, "r") as f:
    data += f.read()

data = findall(r'\w+[-|+|/|.]\w+|<\w+>|\w+', data.lower())
counter = Counter(data)

word_cnt_df =  DataFrame.from_dict(counter, orient = 'index').reset_index()
word_cnt_df.columns = ["word", "count"]

word_cnt_df = word_cnt_df.join(lexicon.set_index("word"), on = "word")
word_cnt_df = word_cnt_df[word_cnt_df["score"].notna()]

sentiment = []

for x in word_cnt_df["score"]:
  if x >= -1.0 and x < -0.6:
    sentiment.append("Neg")
  elif x >= -0.6 and x < -0.2:
    sentiment.append("W.Neg")
  elif x >= -0.2 and x <= 0.2:
    sentiment.append("Neu")
  elif x > 0.2 and x <= 0.6:
    sentiment.append("W.Pos")
  else:
    sentiment.append("Pos")

word_cnt_df["sentiment"] = sentiment

# word_cnt_df = word_cnt_df.groupby(word_cnt_df["sentiment"], as_index = True).agg({"count": "sum"}).agg({"count":"log10"})
word_cnt_df = word_cnt_df.groupby(word_cnt_df["sentiment"], as_index = True).sum()
word_cnt_df["count(log10)"] = log10(word_cnt_df["count"])
word_cnt_df = word_cnt_df.reindex(["Neg", "W.Neg", "Neu", "W.Pos", "Pos"])

plt.title("Series " + series.upper())
plt.xlabel("sentiment")
plt.ylabel("log10 y-axis")
plt.bar(word_cnt_df.index, word_cnt_df["count(log10)"])
plt.savefig('output ' + series.upper())
