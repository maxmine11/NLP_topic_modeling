# Agora's Assignment

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

## Current Approaches to Solution:

### Nonnegative Matrix Factorization (NMF):

#### Approach: 
The NMF technique examines documents and discovers topics in a mathematical framework through probability distributions  depending on your scoring, Kullback-Leibler versus least squares.

#### Advantages:


#### Disadvantages:
* Hard to check for accuracy
#### Algorithm:

Simple implementation straight out of Scikit-learn:
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
 **TOPICS:**




### Latent Dirichlet Distribution (LDA):
#### Approach:
Creates statistical model for discovering abstract topics that occur in a collection of documents. The modeling of are probability distributions over the latent topics and topics are probability distributions over words.  More precisely, NMF with K-L has a Bayesian formalisation as the Gamma-Poisson model.


#### Advantages:
* Documents with similar topics will use similar groups of words.
* Working with probability distributions rather than word frequencies.
* Easy access scope!d to the distribution of words accross topics.

#### Disadvantages:
* Hard to check for accuracy
* Selecting the right number of topics. 
* 
#### Key Questions:

### Probabilistic Latent Sematic Indexing (PLSI):

#### Approach:

#### Advantages:

#### Disadvantages:

#### Key Questions:


## Best Method:

#### Why is this method better than the others?

#### Can I improve this method even better?

#### Have you considered the possibility of improving other methods and comparing them to this one?

#### Can you merge this method with another improve accuracy?

### If you have finalized your decision, get this ship sailing homes!


## Questions to Self:
* Which topics occur in this document?
* Which topics like the i-th word?
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
* Do we need to give a topic a meaningful label or a number representation is enough?

The goal of the model is not to label topics but to get a meaningful representation and compare them in a human like fashion.
* Does the model capture the right aspects of an article?
* What is the distance threshold under which articles are perceived similar?
* How to set a good distance threshold?




