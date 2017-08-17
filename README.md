# Assignment

## Main Task:
#### Given:
* We are given a data set X mapping the index of word "i" to the number of times it appears in document "j" and a data set containing the set of words to index to.
*  From what's given we  want to extract the topics being discussed in New York Times articles. 

#### Expected Result:
*  Each entry of the dataset (below) corresponds to a NYT news article. 
*  The goal is to label each article with a corresponding topic, out of some N total number of topics. 
*  Ideally, the end result is a python file or jupyter notebook that I can run to see the assigned topics.

## Extracting Data Set:
``` python
# File contains index of words appearing in that document,
# and the number of times they appear.
with open('data/nyt_data.txt') as f:
    documents = f.readlines()
documents = [x.strip().strip('\n').strip("'") for x in documents] 

# File contains vocabulary. Load into array and interpret
# vocab[i] = someword, meaning someword has integer id i.
# (i.e. integer id given by location in array)
with open('data/nyt_vocab.dat') as f:
    vocab = f.readlines()
vocab = [w.strip().strip('\n').strip("'") for w in vocab] 

num_docs = 8447
num_words = 3012 
X = np.zeros([num_words, num_docs])

for col in range(len(documents)):
    for row in documents[col].split(','):
        X[int(row.split(':')[0])-1,col] = int(row.split(':')[1])

```

## Relevant Terminology:
* **Stop Words**: Words to filter out in a doc, not always a great idea though.
* **Term frequency**: uses the raw count of a term in a document, i.e. the number of times that term t occurs in document d.


* **Inverse document frequency factor**: incorporated b/c diminishes the weight of terms that occur very frequently a document set and increases the weight of terms that occur rarely. (Important to search for meaninful context)
* **tf-idf** (term frequency and inverse document frequency): how important a word is to a document in a collection or corpus.
* **Document-Term matrix** (Used for ranking functions)
* **Topic Modeling**: a simple way to analyze large volumes of unlabeled text.

## Available Resourcers, Tools, & Libraries (ARTL):
* Brandon's Notes: <http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf>
* Moorisa's Article: <https://medium.com/towards-data-science/topic-modeling-for-the-new-york-times-news-dataset-1f643e15caac>
* My Notes
* LDA (Python Package): <https://radimrehurek.com/gensim/models/ldamodel.html>
* Holly Scikit-learn!
* GloVe: <https://nlp.stanford.edu/projects/glove/>
* Probabilistic Topic Models: <http://psiexp.ss.uci.edu/research/papers/SteyversGriffithsLSABookFormatted.pdf>
* GraphLab: <https://turi.com/learn/userguide/>
* Outstanding basic explanation for LDA: <https://www.youtube.com/watch?v=3mHy4OSyRf0>
* Introduction to Topic Modeling w/ Python: <http://chdoig.github.io/pytexas2015-topic-modeling/#/>
* Beginners guide for topic modeling with Gensim: https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
* Useful Lecture in topic modeling to get a good grip of some minimization functions: <http://www.columbia.edu/~jwp2128/Teaching/W4721/Spring2017/slides/lecture_4-4-17.pdf>
## Current Approaches to Solution:

### Nonnegative Matrix Factorization (NMF):

#### Approach: 
The NMF technique examines documents and discovers topics in a mathematical framework through probability distributions  depending on your scoring, Kullback-Leibler versus least squares.

#### Algorithm:
**Text data:**
  * Word term frequencies
  * $X_{ij}$ contains the number of times word $i$ appears in document $j$.

**Two Commonly used Objective Functions:**
 * Squared Error Objective Function: $$\sum_i\sum_j\|X−WH\|^2 =  (X_{ij} −(WH)_{ij})^2$$
 * Divergence objective: $$D(X\|WH) = −\sum_i\sum_j[X_{ij} \ln(WH)_{ij} − (WH)_{ij}]$$

Both $W$ and $H$ have non-negative entry values.

The main idea revolves around the following:
* Randomly initialize H and W with nonnegative values.
* Iterate the following, first for all values in H, then all in W:$$H_{kj} \leftarrow H_{kj}\frac{(W^TX)_{kj}}{(W^TWH)_{kj}},$$ $$W_{ik} \leftarrow W_{ik}\frac{(XH^T)_{ik}}{(WHH^T)_{ik}},$$ until the change in $\|X − WH\|^2$ is “small.”


Simple implementation using  Scikit-learn and pandas to display topics chosen.
```python
from sklearn.decomposition import NMF
rank = 25
model = NMF(n_components=25, init='random',max_iter=100, random_state=0)
W = model.fit_transform(X)
H = model.components_
'''normalize each column to sum to zero'''
W_normed = W / np.sum(W,axis=0)

'''for each column of W, list the 10 words 
having the largest weight and show the weight'''
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 120)    
vList = []

for topic in range(rank):
    v = pd.DataFrame(vocab)
    v[1] = W_normed[:,topic].round(6)
    v = v.sort_values([1, 0], ascending=[0,1]).rename(index=int, columns={0: "Topic {}".format(topic+1), 1: "Weight"}).head(10)
    v = v.reset_index(drop=True)
    vList.append(v)
    
for num in [5,10,15,20,25]:
    print('\n',(pd.concat(vList[num-5:num], axis=1)),'\n')

```

Notice that the same can be achived writing your own algorithm using the proper objective function but there are libraries such as scikit-learn that make your job a lot simpler. (Know when and when not to use them).


#### Advantages:

* Easier Implementation than LDA.
* Easy interface to incorporate Pandas for better display of topics.
#### Disadvantages:
* Hard to check for accuracy




### Latent Dirichlet Distribution (LDA):
#### Approach:
Creates statistical model for discovering abstract topics that occur in a collection of documents. The modeling of are probability distributions over the latent topics and topics are probability distributions over words.  More precisely, NMF with K-L has a Bayesian formalisation as the Gamma-Poisson model.

### Algorithm:

LDA assumes the following generative process for each document **w** in a corpus $D$:
1. Choose $N ∼ Poisson(\lambda)$.
2. Choose $\theta ∼ Dir(\alpha)$.
3. For each of the $N$ words $w_n$:
    * Choose a topic $z_n ∼ Multinomial(\theta)$.
    * Choose a word $w_n$ from $\mathbb{P}(w_n |z_n,\beta)$, a multinomial probability conditioned on the topic $z_n$.


Simple implementation using Gensim:
``` python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities


# We need to re-write X (our corpus) from our originally given structure
for_dic = []

for col in range(len(documents)):
    for_dic.append([])
    for row in documents[col].split(','):
        word_index = int(row.split(':')[0])-1
        repete = int(row.split(':')[1])
        for i in range(repete):
            for_dic[col].append(vocab[word_index])
            
# Create new Dictionary of words 
dictionary = corpora.Dictionary(for_dic)
# assign a unique integer id to all words appearing in the corpus
# Mapping of the words with their ids
print (dictionary.token2id)

#New X
corpus = [dictionary.doc2bow(text) for text in for_dic]

# The transformations are standard Python objects, typically initialized by means of a training corpus:
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[X1]
#for doc in corpus_tfidf:
#    print (doc)
# Creating the object for LDA model using gensim library
Lda = models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(corpus_tfidf, num_topics=25, id2word = dictionary, passes=50)

print (ldamodel.print_topics(num_topics=25, num_words=10))

```

#### Advantages:
* Documents with similar topics will use similar groups of words.
* Working with probability distributions rather than word frequencies.
* Easy access scope to the distribution of words accross topics.

#### Disadvantages:
* Hard to check for accuracy
* Selecting the right number of topics. 

#### Key Questions:

### Probabilistic Latent Sematic Indexing (PLSI):

#### Approach:
PLSI algorithm can be viewed as learning LDA using a Dirac  variational distribution instead of a Dirichlet.
(Not so much recommended since this is an specific case of LDA)

### Gibbs Sampling
#### Approach:
Gibbs sampling is the proper Bayesian learning of $\theta$, but in 
practice, people just take the mode of the distribution (to avoid label 
switching!), which is close to the solution of variational-LDA.



## Preferred Method -NMF

#### Why is this method better than the others?
It's not that is better but it's way easier to understand and get a grasp of what's really going behing the curtains of topic modeling.
#### Can I improve this method even better?
Definitely, from my implementation above NMF can be made even better by using a tf-idf in our pipeline when creating the term-matrix. Also, as you get your topics and learn more about their distributions you can adjust for better alphas and betas.
#### Have you considered the possibility of improving other methods and comparing them to this one?
Absolutely, Deep learning can help tons to this algorithms the only downside it makes them more computation costly and a bit complex to understand but this is no biggie, you want to be good dont you?

#### Can you merge this method with another improve accuracy?
Again! Yes, you can acomplish great things by choosing the right building blocks to creat a good model that meets your demands.

### If you have finalized your decision, get this ship sailing homes!


## Questions to Self:
* Which topics occur in this document?
Being the New York Times we can kind of predict which topics would be the most notable as you might have already guess yes the hottest were finance, education, tech, etc ...

* How to check how good my algorithm is using the unsupervised learning algorithms NMF and LDA?
Can use Human in the loop approach. (Very costly so not so appealing)
    * Use word Intrution (Looks for word coherence in topics)
    * Use topic Intrution (Intrude a topic that does not belong to a document and ask if they can identify topic that does not belong there.)
Can use Metrics.
    *    Cosine Similarity (Split documents in half to calculate document-topic relation between the same document and other document)
    *    Size (# of tokens assgined)
    *    Similarity to corpus wide distribution
    *    Locally frequent words
    *    Co-doc coherence

* How to choose a good number of topics / what is it that makes two topics appear similar or completely different?

This is hard to do since if we don't know anything about our data prior to processing we are relying on our algorithm to find those relations in the form of topics. 
* Do we need to give a topic a meaningful label or a number representation is enough?

The goal of the model is not to label topics but to get a meaningful representation and compare them in a human like fashion.

* What is the distance threshold under which articles are perceived similar?
* How to set a good distance threshold?




