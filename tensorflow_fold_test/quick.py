
# coding: utf-8

# # TensorFlow Fold Quick Start
# 
# TensorFlow Fold is a library for turning complicated Python data structures into TensorFlow Tensors.

# In[1]:

# boilerplate
import random
import tensorflow as tf
sess = tf.InteractiveSession()
import tensorflow_fold as td


# The basic elements of Fold are *blocks*. We'll start with some blocks that work on simple data types.

# In[2]:

scalar_block = td.Scalar()
vector3_block = td.Vector(3)


# Blocks are functions with associated input and output types.

# In[3]:

def block_info(block):
    print("%s: %s -> %s" % (block, block.input_type, block.output_type))
    
block_info(scalar_block)
block_info(vector3_block)


# We can use `eval()` to see what a block does with its input:

# In[4]:

scalar_block.eval(42)


# In[5]:

vector3_block.eval([1,2,3])


# Not very exciting. We can compose simple blocks together with `Record`, like so:

# In[6]:

record_block = td.Record({'foo': scalar_block, 'bar': vector3_block})
block_info(record_block)


# We can see that Fold's type system is a bit richer than vanilla TF; we have tuple types! Running a record block does what you'd expect:

# In[7]:

record_block.eval({'foo': 1, 'bar': [5, 7, 9]})


# One useful thing you can do with blocks is wire them up to create pipelines using the `>>` operator, which performs function composition. For example, we can take our two tuple tensors and compose it with `Concat`, like so:

# In[8]:

record2vec_block = record_block >> td.Concat()
record2vec_block.eval({'foo': 1, 'bar': [5, 7, 9]})


# Note that because Python dicts are unordered, Fold always sorts the outputs of a record block by dictionary key. If you want to preserve order you can construct a Record block from an OrderedDict.
# 
# The whole point of Fold is to get your data into TensorFlow; the `Function` block lets you convert a TITO (Tensors In, Tensors Out) function to a block:

# In[9]:

negative_block = record2vec_block >> td.Function(tf.negative)
negative_block.eval({'foo': 1, 'bar': [5, 7, 9]})


# This is all very cute, but where's the beef? Things start to get interesting when our inputs contain sequences of indeterminate length. The `Map` block comes in handy here:

# In[10]:

map_scalars_block = td.Map(td.Scalar())


# There's no TF type for sequences of indeterminate length, but Fold has one:

# In[11]:

block_info(map_scalars_block)


# Right, but you've done the TF [RNN Tutorial](https://www.tensorflow.org/tutorials/recurrent/) and even poked at [seq-to-seq](https://www.tensorflow.org/tutorials/seq2seq/). You're a wizard with [dynamic rnns](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn). What does Fold offer?
# 
# Well, how about jagged arrays?

# In[12]:

jagged_block = td.Map(td.Map(td.Scalar()))
block_info(jagged_block)


# The Fold type system is fully compositional; any block you can create can be composed with `Map` to create a sequence, or `Record` to create a tuple, or both to create sequences of tuples or tuples of sequences:

# In[13]:

seq_of_tuples_block = td.Map(td.Record({'foo': td.Scalar(), 'bar': td.Scalar()}))
seq_of_tuples_block.eval([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}])


# In[14]:

tuple_of_seqs_block = td.Record({'foo': td.Map(td.Scalar()), 'bar': td.Map(td.Scalar())})
tuple_of_seqs_block.eval({'foo': range(3), 'bar': range(7)})


# Most of the time, you'll eventually want to get one or more tensors out of your sequence, for wiring up to your particular learning task. Fold has a bunch of built-in reduction functions for this that do more or less what you'd expect:

# In[15]:

((td.Map(td.Scalar()) >> td.Sum()).eval(range(10)),
 (td.Map(td.Scalar()) >> td.Min()).eval(range(10)),
 (td.Map(td.Scalar()) >> td.Max()).eval(range(10)))
        


# The general form of such functions is `Reduce`:

# In[16]:

(td.Map(td.Scalar()) >> td.Reduce(td.Function(tf.multiply))).eval(range(1,10))


# If the order of operations is important, you should use `Fold` instead of `Reduce` (but if you can use `Reduce` you should, because it will be faster):

# In[17]:

((td.Map(td.Scalar()) >> td.Fold(td.Function(tf.divide), tf.ones([]))).eval(range(1,5)),
 (td.Map(td.Scalar()) >> td.Reduce(td.Function(tf.divide), tf.ones([]))).eval(range(1,5)))  # bad, not associative!


# Now, let's do some learning! This is the part where "magic" happens; if you want a deeper understanding of what's happening here you might want to jump right to our more formal [blocks tutorial](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/blocks.md) or learn more about [running blocks in TensorFlow](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/running.md)

# In[18]:

def reduce_net_block():
    net_block = td.Concat() >> td.FC(20) >> td.FC(1, activation=None) >> td.Function(lambda xs: tf.squeeze(xs, axis=1))
    return td.Map(td.Scalar()) >> td.Reduce(net_block)


# The `reduce_net_block` function creates a block (`net_block`) that contains a two-layer fully connected (FC) network that takes a pair of scalar tensors as input and produces a scalar tensor as output. This network gets applied in a binary tree to reduce a sequence of scalar tensors to a single scalar tensor.
# 
# One thing to notice here is that we are calling [`tf.squeeze`](https://www.tensorflow.org/versions/r1.0/api_docs/python/array_ops/shapes_and_shaping#squeeze) with `axis=1`, even though the Fold output type of `td.FC(1, activation=None)` (and hence the input type of the enclosing `Function` block) is a `TensorType` with shape `(1)`. This is because all Fold blocks actually run on TF tensors with an implicit leading batch dimension, which enables execution via [*dynamic batching*](https://arxiv.org/abs/1702.02181). It is important to bear this in mind when creating `Function` blocks that wrap functions that are not applied elementwise.

# In[19]:

def random_example(fn):
    length = random.randrange(1, 10)
    data = [random.uniform(0,1) for _ in range(length)]
    result = fn(data)
    return data, result


# The `random_example` function generates training data consisting of `(example, fn(example))` pairs, where `example` is a random list of numbers, e.g.:

# In[20]:

random_example(sum)


# In[21]:

random_example(min)


# In[22]:

def train(fn, batch_size=100):
    net_block = reduce_net_block()
    compiler = td.Compiler.create((net_block, td.Scalar()))
    y, y_ = compiler.output_tensors
    loss = tf.nn.l2_loss(y - y_)
    train = tf.train.AdamOptimizer().minimize(loss)
    sess.run(tf.global_variables_initializer())
    validation_fd = compiler.build_feed_dict(random_example(fn) for _ in range(1000))
    for i in range(2000):
        sess.run(train, compiler.build_feed_dict(random_example(fn) for _ in range(batch_size)))
        if i % 100 == 0:
            print(i, sess.run(loss, validation_fd))
    return net_block
                 


# Now we're going to train a neural network to approximate a reduction function of our choosing. Calling `eval()` repeatedly is super-slow and cannot exploit batch-wise parallelism, so we create a [`Compiler`](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/td.md#compiler). See our page on [running blocks in TensorFlow](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/running.md) for more on Compilers and how to use them effectively.

# In[23]:

sum_block = train(sum)


# In[24]:

sum_block.eval([1, 1])


# Breaking news: deep neural network learns to calculate 1 + 1!!!!

# Of course we've done something a little sneaky here by constructing a model that can only represent associative functions and then training it to compute an associative function. The technical term for being sneaky in machine learning is [inductive bias](https://en.wikipedia.org/wiki/Inductive_bias).

# In[25]:

min_block = train(min)


# In[26]:

min_block.eval([2, -1, 4])


# Oh noes! What went wrong? Note that we trained our network to compute `min` on positive numbers; negative numbers are outside of its input distribution.

# In[27]:

min_block.eval([0.3, 0.2, 0.9])


# Well, that's better. What happens if you train the network on negative numbers as well as on positives? What if you only train on short lists and then evaluate the net on long ones? What if you used a `Fold` block instead of a `Reduce`? ...  Happy Folding!
