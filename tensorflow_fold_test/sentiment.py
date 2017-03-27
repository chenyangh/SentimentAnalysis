# coding: utf-8

# # Sentiment Analysis with TreeLSTMs in TensorFlow Fold
# 
# The [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/treebank.html) is a corpus of ~10K one-sentence movie reviews from Rotten Tomatoes. The sentences have been parsed into binary trees with words at the leaves; every sub-tree has a label ranging from 0 (highly negative) to 4 (highly positive); 2 means neutral.
# 
# For example, `(4 (2 Spiderman) (3 ROCKS))` is sentence with two words, corresponding a binary tree with three nodes. The label at the root, for the entire sentence, is `4` (highly positive). The label for the left child, a leaf corresponding to the word `Spiderman`, is `2` (neutral). The label for the right child, a leaf corresponding to the word `ROCKS` is `3` (moderately positive).
# 
# This notebook shows how to use TensorFlow Fold train a model on the treebank using binary TreeLSTMs and [GloVe](http://nlp.stanford.edu/projects/glove/) word embedding vectors, as described in the paper [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/pdf/1503.00075.pdf) by Tai et al. The original [Torch](http://torch.ch) source code for the model, provided by the authors, is available [here](https://github.com/stanfordnlp/treelstm).
# 
# The model illustrates three of the more advanced features of Fold, namely:
# 1. [Compositions](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/blocks.md#wiring-things-together-in-more-complicated-ways) to wire up blocks to form arbitrary directed acyclic graphs
# 2. [Forward Declarations](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/blocks.md#recursion-and-forward-declarations) to create recursive blocks
# 3. [Metrics](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#class-tdmetric) to create models where the size of the output is not fixed, but varies as a function of the input data.

# In[1]:

# boilerplate
import codecs
import functools
import os
import tempfile
import zipfile

from nltk.tokenize import sexpr
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
import tensorflow_fold as td

# ## Get the data
# 
# Begin by fetching the word embedding vectors and treebank sentences.

# In[2]:

data_dir = 'data'

# In[3]:


# In[ ]:

full_glove_path = 'data/glove.840B.300d.txt'

# In[5]:

train_path, dev_path, test_path = (
    'data/trees/train.txt', 'data/trees/dev.txt', 'data/trees/test.txt')

# Filter out words that don't appear in the dataset, since the full dataset is a bit large (5GB). This is purely a performance optimization and has no effect on the final results.

# In[6]:

filtered_glove_path = os.path.join(data_dir, 'filtered_glove_path')


# In[7]:

def filter_glove():
    vocab = set()
    # Download the full set of unlabeled sentences separated by '|'.
    sentence_path = 'data/stanfordSentimentTreebank/SOStr.txt'
    with codecs.open(sentence_path, encoding='utf-8') as f:
        for line in f:
            # Drop the trailing newline and strip backslashes. Split into words.
            vocab.update(line.strip().replace('\\', '').split('|'))
    nread = 0
    nwrote = 0
    with codecs.open(full_glove_path, encoding='utf-8') as f:
        with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
            for line in f:
                nread += 1
                line = line.strip()
                if not line: continue
                if line.split(u' ', 1)[0] in vocab:
                    out.write(line + '\n')
                    nwrote += 1
    print('read %s lines, wrote %s' % (nread, nwrote))


# In[8]:

# filter_glove()

# Load the filtered word embeddings into a matrix and build an dict from words to indices into the matrix. Add a random embedding vector for out-of-vocabulary words.

# In[9]:

def load_embeddings(embedding_path):
    """Loads embedings, returns weight matrix and dict from words to indices."""
    print('loading word embeddings from %s' % embedding_path)
    weight_vectors = []
    word_idx = {}
    with codecs.open(embedding_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            word_idx[word] = len(weight_vectors)
            weight_vectors.append(np.array(vec.split(), dtype=np.float32))
    # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
    # '-RRB-' respectively in the parse-trees.
    word_idx[u'-LRB-'] = word_idx.pop(u'(')
    word_idx[u'-RRB-'] = word_idx.pop(u')')
    # Random embedding vector for unknown words.
    weight_vectors.append(np.random.uniform(
        -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    return np.stack(weight_vectors), word_idx


# In[10]:

weight_matrix, word_idx = load_embeddings(filtered_glove_path)


# Finally, load the treebank data.

# In[11]:

def load_trees(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        # Drop the trailing newline and strip \s.
        trees = [line.strip().replace('\\', '') for line in f]
        print('loaded %s trees from %s' % (len(trees), filename))
        return trees


# In[12]:

train_trees = load_trees(train_path)
dev_trees = load_trees(dev_path)
test_trees = load_trees(test_path)


# ## Build the model
# 
# We want to compute a hidden state vector $h$ for every node in the tree. The hidden state is the input to a linear layer with softmax output for predicting the sentiment label. 
# 
# At the leaves of the tree, words are mapped to word-embedding vectors which serve as the input to a binary tree-LSTM with $0$ for the previous states. At the internal nodes, the LSTM takes $0$ as input, and previous states from its two children. More formally,
# 
# \begin{align}
# h_{word} &= TreeLSTM(Embedding(word), 0, 0) \\
# h_{left, right} &= TreeLSTM(0, h_{left}, h_{right})
# \end{align}
# 
# where $TreeLSTM(x, h_{left}, h_{right})$ is a special kind of LSTM cell that takes two hidden states as inputs, and has a separate forget gate for each of them. Specifically, it is [Tai et al.](http://arxiv.org/pdf/1503.00075.pdf) eqs. 9-14 with $N=2$. One modification here from Tai et al. is that instead of L2 weight regularization, we use recurrent droupout as described in the paper [Recurrent Dropout without Memory Loss](http://arxiv.org/pdf/1603.05118.pdf).
# 
# We can implement $TreeLSTM$ by subclassing the TensorFlow [`BasicLSTMCell`](https://www.tensorflow.org/versions/r1.0/api_docs/python/contrib.rnn/rnn_cells_for_use_with_tensorflow_s_core_rnn_methods#BasicLSTMCell).
# 

# In[13]:

class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """LSTM with two state inputs.

    This is the model described in section 3.2 of 'Improved Semantic
    Representations From Tree-Structured Long Short-Term Memory
    Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
    dropout as described in 'Recurrent Dropout without Memory Loss'
    <http://arxiv.org/pdf/1603.05118.pdf>.
    """

    def __init__(self, num_units, keep_prob=1.0):
        """Initialize the cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          keep_prob: Keep probability for recurrent dropout.
        """
        super(BinaryTreeLSTMCell, self).__init__(num_units)
        self._keep_prob = keep_prob

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            lhs, rhs = state
            c0, h0 = lhs
            c1, h1 = rhs
            concat = tf.contrib.layers.linear(
                tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            j = self._activation(j)
            if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
                j = tf.nn.dropout(j, self._keep_prob)

            new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) +
                     c1 * tf.sigmoid(f1 + self._forget_bias) +
                     tf.sigmoid(i) * j)
            new_h = self._activation(new_c) * tf.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


# Use a placeholder for the dropout keep probability, with a default of 1 (for eval).

# In[14]:

keep_prob_ph = tf.placeholder_with_default(1.0, [])

# Create the LSTM cell for our model. In addition to recurrent dropout, apply dropout to inputs and outputs, using TF's build-in dropout wrapper. Put the LSTM cell inside of a [`td.ScopedLayer`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#class-tdscopedlayer) in order to manage variable scoping. This ensures that our LSTM's variables are encapsulated from the rest of the graph and get created exactly once.
# 

# In[15]:

lstm_num_units = 300  # Tai et al. used 150, but our regularization strategy is more effective
tree_lstm = td.ScopedLayer(
    tf.contrib.rnn.DropoutWrapper(
        BinaryTreeLSTMCell(lstm_num_units, keep_prob=keep_prob_ph),
        input_keep_prob=keep_prob_ph, output_keep_prob=keep_prob_ph),
    name_or_scope='tree_lstm')

# Create the output layer using [`td.FC`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#class-tdfc).

# In[16]:

NUM_CLASSES = 5  # number of distinct sentiment labels
output_layer = td.FC(NUM_CLASSES, activation=None, name='output_layer')

# Create the word embedding using [`td.Embedding`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#class-tdembedding). Note that the built-in Fold layers like `Embedding` and `FC` manage variable scoping automatically, so there is no need to put them inside scoped layers.

# In[17]:

word_embedding = td.Embedding(
    *weight_matrix.shape, initializer=weight_matrix, name='word_embedding')

# We now have layers that encapsulate all of the trainable variables for our model. The next step is to create the Fold blocks that define how inputs (s-expressions encoded as strings) get processed and used to make predictions. Naturally this requires a recursive model, which we handle in Fold using a [forward declaration](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/blocks.md#recursion-and-forward-declarations). The recursive step is to take a subtree (represented as a string) and convert it into a hidden state vector (the LSTM state), thus embedding it in a $n$-dimensional space (where here $n=300$).

# In[18]:

embed_subtree = td.ForwardDeclaration(name='embed_subtree')


# The core the model is a block that takes as input a list of tokens. The tokens will be either:
# 
# * `[word]` - a leaf with a single word, the base-case for the recursion, or
# * `[lhs, rhs]` - an internal node consisting of a pair of sub-expressions
# 
# The outputs of the block will be a pair consisting of logits (the prediction) and the LSTM state.

# In[19]:

def logits_and_state():
    """Creates a block that goes from tokens to (logits, state) tuples."""
    unknown_idx = len(word_idx)
    lookup_word = lambda word: word_idx.get(word, unknown_idx)

    word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                td.Scalar('int32') >> word_embedding)

    pair2vec = (embed_subtree(), embed_subtree())

    # Trees are binary, so the tree layer takes two states as its input_state.
    zero_state = td.Zeros((tree_lstm.state_size,) * 2)
    # Input is a word vector.
    zero_inp = td.Zeros(word_embedding.output_type.shape[0])

    word_case = td.AllOf(word2vec, zero_state)
    pair_case = td.AllOf(zero_inp, pair2vec)

    tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

    return tree2vec >> tree_lstm >> (output_layer, td.Identity())


# Note that we use the call operator `()` to create blocks that reference the `embed_subtree` forward declaration, for the recursive case.

# Define a per-node loss function for training.

# In[20]:

def tf_node_loss(logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)


# Additionally calculate fine-grained and binary hits (i.e. un-normalized accuracy) for evals. Fine-grained accuracy is defined over all five class labels and will be calculated for all labels, whereas binary accuracy is defined of negative vs. positive classification and will not be calcluated for neutral labels.

# In[21]:

def tf_fine_grained_hits(logits, labels):
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    return tf.cast(tf.equal(predictions, labels), tf.float64)


# In[22]:

def tf_binary_hits(logits, labels):
    softmax = tf.nn.softmax(logits)
    binary_predictions = (softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1])
    binary_labels = labels > 2
    return tf.cast(tf.equal(binary_predictions, binary_labels), tf.float64)


# The [`td.Metric`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#class-tdmetric) block provides a mechaism for accumulating results across sequential and recursive computations without having the thread them through explictly as return values. Metrics are wired up here inside of a [`td.Composition`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/blocks.md#wiring-things-together-in-more-complicated-ways) block, which allows us to explicitly specify the inputs of sub-blocks with calls to `Block.reads()` inside of a [`Composition.scope()`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#tdcompositionscope) context manager.
# 
# For training, we will sum the loss over all nodes. But for evals, we would like to separately calcluate accuracies for the root (i.e. entire sentences) to match the numbers presented in the literature. We also need to distinguish between neutral and non-neutral sentiment labels, because binary sentiment doesn't get calculated for neutral nodes.
# 
# This is easy to do by putting our block creation code for calculating metrics inside of a function and passing it indicators. Note that this needs to be done in Python-land, because we can't inspect the contents of a tensor inside of Fold (since it hasn't been run yet).

# In[23]:

def add_metrics(is_root, is_neutral):
    """A block that adds metrics for loss and hits; output is the LSTM state."""
    c = td.Composition(
        name='predict(is_root=%s, is_neutral=%s)' % (is_root, is_neutral))
    with c.scope():
        # destructure the input; (labels, (logits, state))
        labels = c.input[0]
        logits = td.GetItem(0).reads(c.input[1])
        state = td.GetItem(1).reads(c.input[1])

        # calculate loss
        loss = td.Function(tf_node_loss)
        td.Metric('all_loss').reads(loss.reads(logits, labels))
        if is_root: td.Metric('root_loss').reads(loss)

        # calculate fine-grained hits
        hits = td.Function(tf_fine_grained_hits)
        td.Metric('all_hits').reads(hits.reads(logits, labels))
        if is_root: td.Metric('root_hits').reads(hits)

        # calculate binary hits, if the label is not neutral
        if not is_neutral:
            binary_hits = td.Function(tf_binary_hits).reads(logits, labels)
            td.Metric('all_binary_hits').reads(binary_hits)
            if is_root: td.Metric('root_binary_hits').reads(binary_hits)

        # output the state, which will be read by our by parent's LSTM cell
        c.output.reads(state)
    return c


# Use [NLTK](http://www.nltk.org/) to define a `tokenize` function to split S-exprs into left and right parts. We need this to run our `logits_and_state()` block since it expects to be passed a list of tokens and our raw input is strings.

# In[24]:

def tokenize(s):
    label, phrase = s[1:-1].split(None, 1)
    return label, sexpr.sexpr_tokenize(phrase)


# Try it out.

# In[25]:

tokenize('(X Y)')

# In[26]:

tokenize('(X Y Z)')


# Embed trees (represented as strings) by tokenizing and piping (`>>`) to `label_and_logits`, distinguishing between neutral and non-neutral labels. We don't know here whether or not we are the root node (since this is a recursive computation), so that gets threaded through as an indicator.

# In[27]:

def embed_tree(logits_and_state, is_root):
    """Creates a block that embeds trees; output is tree LSTM state."""
    return td.InputTransform(tokenize) >> td.OneOf(
        key_fn=lambda pair: pair[0] == '2',  # label 2 means neutral
        case_blocks=(add_metrics(is_root, is_neutral=False),
                     add_metrics(is_root, is_neutral=True)),
        pre_block=(td.Scalar('int32'), logits_and_state))


# Put everything together and create our top-level (i.e. root) model. It is rather simple.

# In[28]:

model = embed_tree(logits_and_state(), is_root=True)

# Resolve the forward declaration for embedding subtrees (the non-root case) with a second call to `embed_tree`.

# In[29]:

embed_subtree.resolve_to(embed_tree(logits_and_state(), is_root=False))

# [Compile](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/running.md#batching-inputs) the model.

# In[30]:

compiler = td.Compiler.create(model)
print('input type: %s' % model.input_type)
print('output type: %s' % model.output_type)

# ## Setup for training
# 
# Calculate means by summing the raw metrics.

# In[31]:

metrics = {k: tf.reduce_mean(v) for k, v in compiler.metric_tensors.items()}

# Magic numbers.

# In[32]:

LEARNING_RATE = 0.05
KEEP_PROB = 0.75
BATCH_SIZE = 100
EPOCHS = 20
EMBEDDING_LEARNING_RATE_FACTOR = 0.1

# Training with [Adagrad](https://www.tensorflow.org/versions/master/api_docs/python/train/optimizers#AdagradOptimizer).

# In[33]:

train_feed_dict = {keep_prob_ph: KEEP_PROB}
loss = tf.reduce_sum(compiler.metric_tensors['all_loss'])
opt = tf.train.AdagradOptimizer(LEARNING_RATE)

# Important detail from section 5.3 of [Tai et al.]((http://arxiv.org/pdf/1503.00075.pdf); downscale the gradients for the word embedding vectors 10x otherwise we overfit horribly.
# 

# In[34]:

grads_and_vars = opt.compute_gradients(loss)
found = 0
for i, (grad, var) in enumerate(grads_and_vars):
    if var == word_embedding.weights:
        found += 1
        grad = tf.scalar_mul(EMBEDDING_LEARNING_RATE_FACTOR, grad)
        grads_and_vars[i] = (grad, var)
assert found == 1  # internal consistency check
train = opt.apply_gradients(grads_and_vars)
saver = tf.train.Saver()

# The TF graph is now complete; initialize the variables.

# In[35]:

sess.run(tf.global_variables_initializer())


# ## Train the model

# Start by defining a function that does a single step of training on a batch and returns the loss.

# In[36]:

def train_step(batch):
    train_feed_dict[compiler.loom_input_tensor] = batch
    _, batch_loss = sess.run([train, loss], train_feed_dict)
    return batch_loss


# Now similarly for an entire epoch of training.

# In[37]:

def train_epoch(train_set):
    return sum(train_step(batch) for batch in td.group_by_batches(train_set, BATCH_SIZE))


# Use [`Compiler.build_loom_inputs()`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#tdcompilerbuild_loom_inputsexamples-metric_labelsfalse-chunk_size100-orderedfalse) to transform `train_trees` into individual loom inputs (i.e. wiring diagrams) that we can use to actually run the model.

# In[38]:

train_set = compiler.build_loom_inputs(train_trees)

# Use [`Compiler.build_feed_dict()`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#tdcompilerbuild_feed_dictexamples-batch_sizenone-metric_labelsfalse-orderedfalse) to build a feed dictionary for validation on the dev set. This is marginally faster and more convenient than calling `build_loom_inputs`. We used `build_loom_inputs` on the train set so that we can shuffle the individual wiring diagrams into different batches for each epoch.

# In[39]:

dev_feed_dict = compiler.build_feed_dict(dev_trees)


# Define a function to do an eval on the dev set and pretty-print some stats, returning accuracy on the dev set.

# In[40]:

def dev_eval(epoch, train_loss):
    dev_metrics = sess.run(metrics, dev_feed_dict)
    dev_loss = dev_metrics['all_loss']
    dev_accuracy = ['%s: %.2f' % (k, v * 100) for k, v in
                    sorted(dev_metrics.items()) if k.endswith('hits')]
    print('epoch:%4d, train_loss: %.3e, dev_loss_avg: %.3e, dev_accuracy:\n  [%s]'
          % (epoch, train_loss, dev_loss, ' '.join(dev_accuracy)))
    return dev_metrics['root_hits']


# Run the main training loop, saving the model after each epoch if it has the best accuracy on the dev set. Use the [`td.epochs`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#tdepochsitems-nnone-shuffletrue-prngnone) utility function to memoize the loom inputs and shuffle them after every epoch of training.

# In[41]:

best_accuracy = 0.0
save_path = os.path.join(data_dir, 'sentiment_model')
for epoch, shuffled in enumerate(td.epochs(train_set, EPOCHS), 1):
    train_loss = train_epoch(shuffled)
    accuracy = dev_eval(epoch, train_loss)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        checkpoint_path = saver.save(sess, save_path, global_step=epoch)
        print('model saved in file: %s' % checkpoint_path)

# The model starts to overfit pretty quickly even with dropout, as the LSTM begins to memorize the training set (which is rather small).

# ## Evaluate the model
# 
# Restore the model from the last checkpoint, where we saw the best accuracy on the dev set.

# In[42]:

saver.restore(sess, checkpoint_path)

# See how we did.

# In[43]:

test_results = sorted(sess.run(metrics, compiler.build_feed_dict(test_trees)).items())
print('    loss: [%s]' % ' '.join(
    '%s: %.3e' % (name.rsplit('_', 1)[0], v)
    for name, v in test_results if name.endswith('_loss')))
print('accuracy: [%s]' % ' '.join(
    '%s: %.2f' % (name.rsplit('_', 1)[0], v * 100)
    for name, v in test_results if name.endswith('_hits')))


# Not bad! See section 3.5.1 of [our paper](https://arxiv.org/abs/1702.02181) for discussion and a comparison of these results to the state of the art.
