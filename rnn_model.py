import tensorflow as tf
import itertools as it
import os
import numpy as np

class Datahandle:

    def __init__(self, path):
        self.path = path
    
    def get_files(self):
        files = os.listdir(self.path)
        for f in files:
            with open(os.path.join(self.path, f), 'r') as myfile:
                txt = myfile.read().replace('\n', '')
            yield txt
    
    def process_raw(self):
        txt_gen = self.get_files()
        chars = []
        unique = set() 
        for txt in txt_gen:
            cs = list(txt)
            chars.extend(cs)
            unique.update(cs)
        self.chars = chars
        self.unique = list(unique)
        self.build_dictionaries()

    def build_dictionaries(self):
        self.idx_to_char = {idx:c for idx, c in enumerate(self.unique)}
        self.char_to_idx = {c:idx for idx, c in enumerate(self.unique)}
        
    def batch_gen(self, batch_size, seq_len):
        def seq_gen():
            l = len(self.chars)
            max_idx = l - seq_len
            list_idx = np.arange(0, max_idx, 100)
            np.random.shuffle(list_idx)
            for step in list_idx:
                seqX = self.chars[step:step + seq_len]
                seqY = self.chars[step + 1:step + 1 + seq_len]
                seqX = [self.char_to_idx[s] for s in seqX]
                seqY = [self.char_to_idx[s] for s in seqY]
                yield (seqX, seqY)
        SG = seq_gen()
        while(True):
            batch = list(it.islice(SG, 0, batch_size))
            batchX = np.array([b[0] for b in batch])
            batchY = np.array([b[1] for b in batch])
            if (len(batchX.shape) < 2 or batchX.shape[0] < batch_size):
                SG = seq_gen()
                print()
                print('Finish')
                print()
                continue
            yield (batchX, batchY)
    

class RNNmodel:
    def __init__(self, batch_size, seq_len):
        print('new')
        self.batch_size = batch_size
        self.seq_len = seq_len

    def get_data(self, path):
        D = Datahandle(path)
        self.D = D
        D.process_raw()
        self.get_batch = D.batch_gen(self.batch_size, self.seq_len)
        j = 0
        #for i in self.get_batch:
            #if(i[0].shape[0] < 50 or i[1].shape[0] < 50):
                #print(i[0].shape)
                #print(i[1].shape)
        #exit() 
        self.num_classes = len(D.unique)
    
    def model(self):
        batch_size = self.batch_size 
        num_classes = self.num_classes

        n_hidden = 700 
        n_layers = 3
        truncated_backprop = self.seq_len 
        dropout = 0.3 
        learning_rate = 0.001
        epochs = 200
        
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [batch_size, truncated_backprop], name='x')
            y = tf.placeholder(tf.int32, [batch_size, truncated_backprop], name='y')
        
        with tf.name_scope('weights'):
            W = tf.Variable(np.random.rand(n_hidden, num_classes), dtype=tf.float32)
            b = tf.Variable(np.random.rand(1, num_classes), dtype=tf.float32)

        inputs_series = tf.split(x, truncated_backprop, 1)
        labels_series = tf.unstack(y, axis=1)
        
        with tf.name_scope('LSTM'):
            cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layers)
        
        states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, \
            dtype=tf.float32)
       
        logits_series = [tf.matmul(state, W) + b for state in states_series]
        prediction_series = [tf.nn.softmax(logits) for logits in logits_series]
        
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) \
            for logits, labels, in zip(logits_series, labels_series)]
        total_loss = tf.reduce_mean(losses)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        
        tf.summary.scalar('total_loss', total_loss)
        summary_op = tf.summary.merge_all()
        
        loss_list = []
        writer = tf.summary.FileWriter('tf_logs', graph=tf.get_default_graph())
        
        all_saver = tf.train.Saver()
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #tf.reset_default_graph()
            #saver = tf.train.import_meta_graph('./models/tf_models/rnn_model.meta')
            #saver.restore(sess, './models/tf_models/rnn_model')

            for epoch_idx in range(epochs):
                xx, yy = next(self.get_batch)
                batch_count = len(self.D.chars) // batch_size // truncated_backprop

                for batch_idx in range(batch_count):
                    batchX, batchY = next(self.get_batch)
   
                    summ, _total_loss, _train_step, _current_state, _prediction_series = sess.run(\
                        [summary_op, total_loss, train_step, current_state, prediction_series],
                        feed_dict = {
                            x : batchX,
                            y : batchY
                        })

                    loss_list.append(_total_loss)
                    writer.add_summary(summ, epoch_idx * batch_count + batch_idx)
                    if batch_idx % 5 == 0:
                        print('Step', batch_idx, 'Batch_loss', _total_loss)
                    
                    if batch_idx % 50 == 0:
                        all_saver.save(sess, 'models/tf_models/rnn_model')

                if epoch_idx % 5 == 0:
                    print('Epoch', epoch_idx, 'Last_loss', loss_list[-1])

rnn = RNNmodel(batch_size=200, seq_len=100)

rnn.get_data('data/HPserie')

rnn.model()
