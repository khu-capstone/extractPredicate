import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from lib.word2idx import Word2idx

def getData():
  df = getDataFromFile()

  x_data = df.drop(['predicate'], axis = 1)
  y_data = outputMapping(df['predicate'])

  return x_data, y_data

def getDataFromFile():
  ol_df = pd.read_csv('data/ol.csv', error_bad_lines=False)
  ul_df = pd.read_csv('data/ul.csv', error_bad_lines=False)
  ol_df.insert(1, 'ol', 1)
  ul_df.insert(1, 'ol', 0)
  df = pd.concat([ol_df, ul_df])
  df = df.dropna()
  df['predicate'] = df['predicate'].str.strip()
  df['predicate'] = df['predicate'].str.replace('"', '').replace("'", '')
  return df


# output 설정
# is = 0
# Nth S is = 1
# have = 2
# other = 3
def outputMapping(output):
  class cls(dict):
    def __missing__(self, key):
      return 3
  return output.map(cls({ 'is': 0, 'Nth S is': 1, 'have': 2 })).astype(int)