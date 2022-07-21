# Training-a-word2vec-model




Word Embedding:
Word embeddings in short are vector representation of words.In this model each word in our corpus will be mapped to real valued vector. The main idea of using these word emebeddings is to capture the semantic/syntactical information possible.
 Traditionally when we consider natural language processing systems, they consider words as atomic symbols. Which means practically they will assign a ID to each word. For example for the word rainbow it assigns ‘ID 123’ and for rain it assigns ‘ID456’. So from the above ID we will not get either information about rainbow or rains or relationship between them. From the above method our model will not learn anything. By using word embeddings we can overcome this problem, as they have the ability analyse them semantically and syntactically. We can also define it as the distance between the words define the similarity between those words. Using these word embeddings will give great knowledge to our model as we are working on semantic analysis.
This work will be introducing the below state of the art word embedding methods. They are
Word2vec
Word2vec is the first neural embedding model that concentrates on target words which is different to previous methods. The target words are represented by surrounding words, with a neural network which encodes the representations.
The basic idea of neural network word embeddings to is to predict the target word Wt and context words in terms of word vectors
P(Context/Wt) with loss function J= 1-P(W-t/Wt).
We look at many positions ’t’ in big corpus. We keep modifying our vector representation to minimize the loss. Then we can visualize our vectors in 2d space and we can clearly observe that vectors captured information about words and their relationships to others which useful in semantic way. Ex: Information related Male-Female,Different tenses of verbs and even country-capital relationships.
 There are two methods of word2vec –Skip gram & Continuous bag of words (CBOW) which are discussed briefly below.
CBOW:
CBOW is a neural network embedding model where it takes context of each word as input and tries to predict the target words which are likely to appear. For example in the sentence ‘Have a great day’, considering input as ‘great’ and tries to predict the context words.
In detailed, the input and output layers are one hot encoded with size [1*V]. 
Now there will be two weights which are between input and hidden layer [V*N] and one between hidden layer and output layer [N*V] and N denotes the number of dimensions.
The hidden input and hidden output are multiplied by which we can get output. 
Error between output, target is calculated and returned back for re adjusting.
The weight between hidden layer and output layer is considered as word-vector representation.
This is explained for a single word context. For multiple word context it looks like below in the diagram.

SKIP Gram:
This model will also have same topology as the previous model with just a flip in architecture. It tends to works the other way round to CBOW. The difference between CBOW and Skip gram is the way they generate word to vectors. Skip gram model predicts the context given a word. The working model is shown below in the diagram.
Glove
Glove denotes to global vectors. This model captures the meaning of one word embedding based on word frequency and co-occurrence counts of whole corpus. It means given a corpus, it builds a co-occurrence matrix of words in vocabulary in a given context size. It learns by minimizing the cost function J, to get our word embeddings.
Learning: J
Fasttext.
Fasttext enriches word vectors with sub word information. It is an extension of the previously mentioned Word2vec-Skipgram model. This model allows creating an unsupervised/supervised learning for obtaining word embeddings. 

Normally fastetxt takes internal structure of a word into account and divides it into n-grams to get those embeddings. Later they will be added to get the final embedding of that word. For example let us consider the word ‘college’ and the trigrams from it would be ‘col’,’oll’,’lle’,’leg’. The word embedding for the word college will be the sum of all these trigrams.
Fasttext allows getting the embeddings of words which are out of vocabulary. But due the calculation of n-grams and embeddings it takes a bit more computing time than other embedding models. Normally these n-grams can be of any size, but the ideal value of n is from 3-6.
