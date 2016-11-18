"""
Method: recurrent neural net
Editor: Thanh L.X.
"""

from __future__ import print_function
import tensorflow as tf
from data import processData

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("summaries_dir", "/tmp/lstm_logs", "directory to summaries")
flags.DEFINE_float("starter_learning_rate", 0.1, "starter learning rate")
flags.DEFINE_integer("training_iters", 5000, "number of training iterations")
flags.DEFINE_integer("batch_size", 50, "size of batch")
flags.DEFINE_integer("display_step", 10, "number of steps ==> display")
flags.DEFINE_integer("seq_max_len", 20, "sequence max length")
flags.DEFINE_integer("n_hidden", 50, "number of hidden neurons")
flags.DEFINE_integer("n_classes", 2, "number of sentiment classes - [0,1]")
flags.DEFINE_boolean("train_mode", True, "training mode is set to True by default")
flags.DEFINE_string("checkpoint_dir", "/tmp/lstm_checkpoint", "directory to checkpoint")


# here we modified split_data function to extract only 100000 samples at most
def split_data(input, split_ratio):
    dataset = []
    count = 0
    with open(input) as infile:
        for line in infile:
            count += 1
            if count <= 100000:
                line = line.strip()
                dataset.append(line)
    if not split_ratio:
        return list(dataset)
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    valid_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(valid_set))
        train_set.append(valid_set.pop(index))
    return [train_set, valid_set]


# ============= #
# DATA PARSER
# ============= #
class SequenceData(object):
    def __init__(self, max_seq_len=200, min_seq_len=1, data_set=None, table=None, dictionary=None, labeled=True):
        self.data = []
        self.labels = []
        self.seqlen = []
        if labeled:
            dim = table.shape[1]
            n_class = 2
            for line in data_set:
                line = line.strip()
                y = line.split("\t")[0]
                x = line.split("\t")[1].split(",")
                length = len(x)
                assert length >= min_seq_len
                zero = [0.] * dim
                if length > max_seq_len:
                    x = x[:max_seq_len]
                    xx = [dictionary.index(i) for i in x]
                    s = [table[i] for i in xx]
                    self.seqlen.append(max_seq_len)
                else:
                    xx = [dictionary.index(i) for i in x]
                    s = [table[i] for i in xx]
                    s += [zero for _ in range(max_seq_len - length)]
                    self.seqlen.append(length)

                self.data.append(s)
                s_label = [0] * n_class
                s_label[int(y)] = 1
                self.labels.append(s_label)
        else:
            dim = table.shape[1]
            n_class = 2
            for line in data_set:
                line = line.strip()
                y = "0"
                x = line.split()
                length = len(x)
                assert length >= min_seq_len
                zero = [0.] * dim
                if length > max_seq_len:
                    x = x[:max_seq_len]
                    xx = [dictionary.index(i) for i in x]
                    s = [table[i] for i in xx]
                    self.seqlen.append(max_seq_len)
                else:
                    xx = [dictionary.index(i) for i in x]
                    s = [table[i] for i in xx]
                    s += [zero for _ in range(max_seq_len - length)]
                    self.seqlen.append(length)
                self.data.append(s)
                s_label = [0] * n_class
                s_label[int(y)] = 1
                self.labels.append(s_label)

        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


# ===================== #
#          MODEL
# ===================== #


# clear the default graph stack and reset the global default graph
print("Reset the global default graph.. ")
tf.reset_default_graph()

# Parameters


# Specify path-to-source
source_path = "./data/"


print("===================")
print("Dynamic RNN")
print("===================")
print("Loading lookup table")

print("================")
print ("Training-validating")

split_ratio = 0.9
source_path = "./data/"
problem_id = "small"
source_train = "training-data-"+problem_id+".txt"

# train_set, valid_set = processData.split_data(source_path + "training-data-"+problem_id+".txt", split_ratio)
train_set, valid_set = processData.split_data(source_path + "training-data-"+problem_id+".txt", split_ratio)
set_dict = processData.to_dict(train_set)
length_dict = len(set_dict)
Table = processData.one_hot_table(length_dict)
print(Table)
print(Table.shape)
# Passing data
print("Passing data")
trainset = SequenceData(max_seq_len=FLAGS.seq_max_len, data_set=train_set,
                        table=Table, dictionary=set_dict)

if FLAGS.train_mode:
    testset = SequenceData(max_seq_len=FLAGS.seq_max_len, data_set=valid_set,
                           table=Table, dictionary=set_dict)
else:
    test_set = processData.split_data(source_path + "test-data-"+problem_id+".txt", False)
    test_set = test_set[:1000]
    test_set = processData.scan(test_set, set_dict)
    testset = SequenceData(max_seq_len=FLAGS.seq_max_len, data_set=test_set,
                           table=Table, dictionary=set_dict, labeled=False)

print("check the data size")
print(len(trainset.data), len(trainset.data[0]), len(trainset.data[0][0]))
print(len(testset.data), len(testset.data[0]), len(testset.data[0][0]))

# tf Graph input
print("check input dimension")
dimension = Table.shape[1]
print(dimension)

print("start training ..")

# Placeholder for data
x = tf.placeholder("float", [None, FLAGS.seq_max_len, dimension])
# Placeholder for label
y = tf.placeholder("float", [None, FLAGS.n_classes])
# A placeholder for indicating each sequence length
seq_len = tf.placeholder(tf.int32, [None])

# Define weights
with tf.name_scope("weights"):
    weights = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_classes]))
    }
with tf.name_scope("biases"):
    biases = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
    }


def dynamic_rnn(x, seqlen, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, dimension])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, FLAGS.seq_max_len, x)
    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.GRUCell(FLAGS.n_hidden)
    # Get lstm cell output, providing 'sequence_length' will perform dynamic calculation.
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * FLAGS.seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, FLAGS.n_hidden]), index)
    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']


# Get output from Dynamic_RNN
pred = dynamic_rnn(x, seq_len, weights, biases)
prediction = tf.argmax(pred, 1)
# Define loss and optimizer
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
tf.scalar_summary("cost", cost)

with tf.name_scope("learning_rate"):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.starter_learning_rate, global_step, 10, 0.96, staircase=True)
    # learning_rate = FLAGS.starter_learning_rate
tf.scalar_summary("learning_rate", learning_rate)


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

# Evaluate model
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary("accuracy", accuracy)

# Merge all the summaries and write them out to /tmp/lstm_logs (by default)
if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)

# make directory /tmp/lstm_checkpoint if does not exist
if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

merged = tf.merge_all_summaries()
sess = tf.InteractiveSession()
train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

# Initializing the variables
init = tf.initialize_all_variables()

# Using defaults to saving all variables
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    if FLAGS.train_mode:
        step = 1
        # Keep training until reach max iterations
        while step * FLAGS.batch_size < FLAGS.training_iters:
            batch_x, batch_y, batch_seqlen = trainset.next(FLAGS.batch_size)
            # Run optimization (back-prop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           seq_len: batch_seqlen})
            if step % FLAGS.display_step == 0:
                # Calculate batch accuracy
                summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_x, y: batch_y, seq_len: batch_seqlen})
                train_writer.add_summary(summary, step)
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seq_len: batch_seqlen})
                print("Iter " + str(step * FLAGS.batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))
                saver.save(sess, FLAGS.checkpoint_dir + "/model.ckpt", global_step=step)
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for Validation set
        test_data = testset.data
        test_label = testset.labels
        test_seq_len = testset.seqlen
        summary, acc = sess.run([merged, accuracy], feed_dict={x: test_data, y: test_label, seq_len: test_seq_len})
        test_writer.add_summary(summary, step)
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                            seq_len: test_seq_len}))
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restore model parameters
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print ("no checkpoint found..")
        # Generate prediction for Test set
        test_data = testset.data
        test_label = testset.labels
        test_seq_len = testset.seqlen
        print("Generate prediction:",
              sess.run(prediction, feed_dict={x: test_data, y: test_label,
                                              seq_len: test_seq_len}))
        print (prediction)

