import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy as np
import theano
import theano.tensor as T
import random

from logistic_sgd import LogisticRegression

from theano.tensor.signal import downsample
from random import shuffle

from loadData import load_SNLI_dataset, load_word2vec_to_init, load_word2vec
from common_functions import create_conv_para,GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit, row_wise_cosine, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, L2norm_paraList, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, L2_weight=0.001, Div_reg=0.001, emb_size=50, hidden_size=50, batch_size=200, maxSentLen=40):
    '''
    epoch: iterating over all training examples once is called one epoch, usually this process will repeated multiple times
    L2_weight: the parameter for L2 normalization;
    Div_reg: the parameter for Diversity normalization;
    emb_size: the dimension of initialized word representations in the beginning;
    hidden_size: the dimension of some hidden states;
    batch_size: how many sentences our model deals with together;
    filter_size: how many consecutive words CNN deal with in one sliding window;
    maxSentLen: to control the model complexity, we truncate all sentences into maximal length
    '''
    model_options = locals().copy()
    print "model options", model_options    
    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results 

    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id  =load_SNLI_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_sents_l=all_sentences_l[0]
    test_sents_l=all_sentences_l[1]

    train_masks_l=all_masks_l[0]
    test_masks_l=all_masks_l[1]

    train_sents_r=all_sentences_r[0]
    test_sents_r=all_sentences_r[1]    
    
    train_masks_r=all_masks_r[0]
    test_masks_r=all_masks_r[1]
        
    train_labels_store=all_labels[0]
    test_labels_store=all_labels[1]
    
    train_size=len(train_labels_store)
    test_size=len(test_labels_store)
    #posi_test_size=np.sum(np.asarray(test_labels_store))
    #print 'posi_test_size:', posi_test_size, 'test_size:', test_size 
    #exit(0) 



    vocab_size=  len(word2id)+1 # add one zero pad index
                    
    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
#     rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
#     id2word = {y:x for x,y in word2id.iteritems()}
#     word2vec=load_word2vec()
#     rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable      
    
    
    #now, start to build the input form of the model
    sentBatch_l=T.imatrix('sentBatch_l')
    sentBatch_r=T.imatrix('sentBatch_r')
    sentBatch_mask_l=T.fmatrix('sentBatch_mask_l')
    sentBatch_mask_r=T.fmatrix('sentBatch_mask_r')
    labels=T.ivector('labels')
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'    
    
    sentBatch_l_input=embeddings[sentBatch_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    sentBatch_r_input=embeddings[sentBatch_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)
    

    
    #GRU
    U, W, b=create_GRU_para(rng, emb_size, hidden_size)
    NN_l_para=[U, W, b]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
    gru_layer_l=GRU_Batch_Tensor_Input_with_Mask(sentBatch_l_input, sentBatch_mask_l,  hidden_size, U, W, b)
    sent_hiddens_l=gru_layer_l.output_tensor  # (batch_size, hidden_size, senlen)
    sent_reps_l=gru_layer_l.output_sent_rep   #(batch, hidden)
 
    U_a, W_a, b_a=create_GRU_para(rng, emb_size, hidden_size)
    NN_a_para=[U_a, W_a, b_a]
    
    sent_hiddens_l=sent_hiddens_l.dimshuffle(0,2,1).reshape((batch_size*maxSentLen, hidden_size))
    sentBatch_r_input_repeat=T.repeat(sentBatch_r_input, maxSentLen, axis=0) #(batch*maxSenLen, hidden, maxSenLen)
    sentBatch_mask_r_repeat=T.repeat(sentBatch_mask_r, maxSentLen, axis=0)
    iter_step=GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit(sentBatch_r_input_repeat, sentBatch_mask_r_repeat, sent_hiddens_l, hidden_size, U_a, W_a, b_a)
    repeat_r_reps=iter_step.output_sent_rep #(batch*maxSenLen, hidden)
    r_tensors=repeat_r_reps.reshape((batch_size, maxSentLen, hidden_size)).dimshuffle(0,2,1) #(batch_size, hidden, maxsenlen)
#     def attention_step(ini_matrix):
#         #ini_matrix (batch, hidden)
#         iter_step=GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit(sentBatch_r_input, sentBatch_mask_r, ini_matrix, hidden_size, U_a, W_a, b_a)
#         return iter_step.output_sent_rep #(batch, hidden)
#  
#     acc_attention_tensor3, updates = theano.scan(
#         attention_step,
#         sequences=sent_hiddens_l.dimshuffle(2,0,1))  #(sentlen, batch, hidden)
#      
    U_r, W_r, b_r=create_GRU_para(rng, hidden_size, hidden_size)
    NN_r_para=[U_r, W_r, b_r]
    gru_layer_r=GRU_Batch_Tensor_Input_with_Mask(r_tensors, sentBatch_mask_l,  hidden_size, U_r, W_r, b_r)
    sent_reps_r=gru_layer_r.output_sent_rep
#     sent_reps_l=T.sum(sentBatch_l_input, axis=2)
#     sent_reps_r=T.sum(sentBatch_r_input, axis=2)
    
    LR_input=T.concatenate([sent_reps_l, sent_reps_r, row_wise_cosine(sent_reps_l, sent_reps_r)], axis=1)
     
    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    U_LR = create_ensemble_para(rng, 3, 2*hidden_size+1) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_LR, LR_b]
    layer_LR=LogisticRegression(rng, input=LR_input, n_in=hidden_size*2Z, n_out=3, W=U_LR, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    
    params = [embeddings]+NN_l_para+NN_a_para+NN_r_para+LR_para   # put all model parameters together
    L2_reg =L2norm_paraList([embeddings,U, W, U_a, W_a, U_r, W_r, U_LR])
    diversify_reg= Diversify_Reg(U_LR.T)#+Diversify_Reg(conv_W_into_matrix)

    cost=loss#+L2_weight*L2_reg
    
    grads = T.grad(cost, params)    # create a list of gradients for all model parameters
    '''
    #implement AdaGrad for updating NN. Traditional parameter updating rule is: P_new=P_old - learning_rate*gradient.
    AdaGrad is an improved version of this, it changes the gradient (you can also think it changes the learning rate) by considering all historical gradients
    In below, "accumulator" is used to store the accumulated history gradient for each parameter.
    '''
    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))    


    '''
    for a theano function, you just need to tell it what are the inputs, what is the output. In below, "sents_id_matrix, sents_mask, labels" are three inputs, you put them
    into a list, "cost" is the output of the training model; "layer_LR.errors(labels)" is the output of test model as we are interested in the classification accuracy of 
    test data. This kind of error will be changed into accuracy afterwards
    '''
    train_model = theano.function([sentBatch_l, sentBatch_r, sentBatch_mask_l, sentBatch_mask_r, labels], cost, updates=updates,on_unused_input='ignore')
    
    test_model = theano.function([sentBatch_l, sentBatch_r, sentBatch_mask_l, sentBatch_mask_r, labels], layer_LR.errors(labels), on_unused_input='ignore')    
    
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False
    
    '''
    split training/test sets into a list of mini-batches, each batch contains batch_size of sentences
    usually there remain some sentences that are fewer than a normal batch, we can start from the "train_size-batch_size" to the last sentence to form a mini-batch
    or cource this means a few sentences will be trained more times than normal, but doesn't matter
    '''
    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]

        
    max_acc=0.0
    combined=range(train_size)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #combined = zip(train_paths_store,train_targets_store,train_masks_store,train_labels_store)
        
        random.shuffle(combined) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            
            batch_indices=combined[batch_id:batch_id+batch_size]
            cost_i+= train_model(
                                np.asarray([train_sents_l[id] for id in batch_indices], dtype='int32'), 
                                      np.asarray([train_sents_r[id] for id in batch_indices], dtype='int32'), 
                                      np.asarray([train_masks_l[id] for id in batch_indices],dtype=theano.config.floatX),
                                      np.asarray([train_masks_r[id] for id in batch_indices], dtype=theano.config.floatX),
                                      np.asarray([train_labels_store[id] for id in batch_indices], dtype='int32'))

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.time()
                  
                error_sum=0.0
                for test_batch_id in test_batch_start: # for each test batch
                    

                    error_i=test_model(
                                np.asarray(test_sents_l[test_batch_id:test_batch_id+batch_size], dtype='int32'), 
                                      np.asarray(test_sents_r[test_batch_id:test_batch_id+batch_size],dtype='int32'),
                                      np.asarray(test_masks_l[test_batch_id:test_batch_id+batch_size], dtype=theano.config.floatX),
                                      np.asarray(test_masks_r[test_batch_id:test_batch_id+batch_size], dtype=theano.config.floatX),
                                      np.asarray(test_labels_store[test_batch_id:test_batch_id+batch_size], dtype='int32'))
                    
                    error_sum+=error_i
                accuracy=1.0-error_sum/(len(test_batch_start))
                if accuracy > max_acc:
                    max_acc=accuracy
                print 'current acc:', accuracy, '\t\t\t\t\tmax acc:', max_acc

                        



            if patience <= iter:
                done_looping = True
                break
        
        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()
            
        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                    
                    
                    
                    
                    
if __name__ == '__main__':
    evaluate_lenet5()
