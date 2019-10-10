# Importing the libraries
import pandas as pd
import re
import nltk
import gensim
import gensim.models as gm


df = pd.read_csv('Jokes.csv')

x=df['question'].values.tolist()
y=df['answer'].values.tolist()

corpus=x+y

tok_corp=[nltk.word_tokenize(sent.decode('utf=8'))for sent in corpus]
model= gensim.models.word2vec(tok_corp,min_count=1,size=32)


model.save('Jokesmodel')  # stores only syn0, not syn0norm
model = gensim.models.word2vec.load('Jokesmodel')  # mmap the large matrix as read-only
model.most_similar('word')   #to get the similar word
model['word']   #to get the vector representation of a word








###########------alternate using pre processing----------------------############

# Importing the libraries
import pandas as pd
import re
import nltk
import gensim
import gensim.models as gm


df = pd.read_csv('Jokes.csv')
# Cleaning the texts
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(1, 118):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Description'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    x=df['question'].values.tolist()
    y=df['answer'].values.tolist()
    
    corpus=x+y
    
tok_corp=[nltk.word_tokenize(sent.decode('utf=8'))for sent in corpus]
model= gensim.models.word2vec(tok_corp,min_count=1,size=32)
    
    


 