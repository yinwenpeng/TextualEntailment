import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN

from logistic_sgd import LogisticRegression
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from operator import itemgetter

def load_ibm_corpus(vocabFile, trainFile, devFile, maxlength):
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    sentlength_limit=1040
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # label, question, answer
            #question
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().split()  
                length=len(words)
                if length>sentlength_limit:
                    words=words[:sentlength_limit]
                    length=sentlength_limit
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent+=[0]*left
                for word in words:
                    sent.append(word2id.get(word))
                sent+=[0]*right
                data.append(sent)
                del sent
                del words
            line_control+=1
            if line_control%100==0:
                print line_control
        read_file.close()
        return numpy.array(data),numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_dev_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().split() 
                length=len(words)
                if length>sentlength_limit:
                    words=words[:sentlength_limit]
                    length=sentlength_limit
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)  
                sent+=[0]*left
                for word in words:
                    sent.append(word2id.get(word))
                sent+=[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==1000:
            #    break
        read_file.close()
        return numpy.array(data),Y, numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_file(devFile, vocab)
    print 'dev file loaded over, total pairs: ', len(devLengths)/2
   

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y


    train_set_Lengths=shared_dataset(trainLengths)                             
    valid_set_Lengths = shared_dataset(devLengths)
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    dev_left_pad=shared_dataset(devLeftPad)
    dev_right_pad=shared_dataset(devRightPad)
                                
    #valid_set_y = shared_dataset(devY)
    

    rval = [(indices_train,train_set_Lengths, train_left_pad, train_right_pad), (indices_dev, devY, valid_set_Lengths, dev_left_pad, dev_right_pad)]
    return rval, word_ind-1

def load_word2vec_to_init(rand_values, ivocab, word2vec):
    
    for id, word in ivocab.iteritems():
        emb=word2vec.get(word)
        if emb is not None:
            rand_values[id]=numpy.array(emb)
    print '==> use word2vec initialization over...'
    return rand_values
    
def load_msr_corpus(vocabFile, trainFile, testFile, maxlength): #maxSentLength=60
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # label, sent1, sent2
            Y.append(int(tokens[0])) #repeat
            Y.append(int(tokens[0])) 
            #question
            for i in [1,2,2,1]: #shuffle the example
                sent=[]
                words=tokens[i].strip().lower().split()  
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1

                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #line_control+=1
        read_file.close()
        '''
        #normalized length
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            #Y.append(int(tokens[0]))
            for i in [1,2]:
                sent=[]
                words=tokens[i].strip().lower().split()  
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1

                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #line_control+=1
            #if line_control==1000:
            #    break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')
        #return shared_y


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
                                
    train_set_y=shared_dataset(trainY)                             
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1

def load_mts(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
        train_values.append(tokens)#repeat once
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values

def load_mts_wikiQA(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values

def load_extra_features(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values
def load_wmf_wikiQA(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values
def load_wikiQA_corpus(vocabFile, trainFile, testFile, max_truncate,maxlength): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            Y.append(int(tokens[2])) 
            #question
            for i in [0,1]:
                sent=[]
                words=tokens[i].strip().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    #print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==50:
            #    break
        read_file.close()
        '''
        #normalized length
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[2])) # make the label starts from 0 to 4
            #Y.append(int(tokens[0]))
            for i in [0,1]:
                sent=[]
                words=tokens[i].strip().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    #print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right) 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #line_control+=1
            #if line_control==1000:
            #    break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')  # for ARC-II on gpu
        #return shared_y


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
                                
    train_set_y=shared_dataset(trainY)                             
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1


def load_entailment_corpus(vocabFile, trainFile, testFile, max_truncate,maxlength): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            question=tokens[1].strip().lower().split() 
            answer=tokens[2].strip().lower().split()   
            if len(question)>max_truncate or len(answer)>max_truncate or len(question)< 2 or len(answer)<2:
                continue #skip this pair
            else:
                Y.append(int(tokens[0])) 
                sents=[question, answer]
                #question
                for i in [0,1]:
                    sent=[]
                    words=sents[i]
                    #true_lengths.append(len(words))
                    length=0
                    for word in words:
                        id=word2id.get(word)
                        if id is not None:
                            sent.append(id)
                            length+=1
                            #if length==max_truncate: #we consider max 43 words
                            #    break
                    if length==0:
                        print 'shit sentence: ', sents[i]
                        #exit(0)
                        break
                    Lengths.append(length)
                    left=(maxlength-length)/2
                    right=maxlength-left-length
                    leftPad.append(left)
                    rightPad.append(right)
     
                    sent=[0]*left+sent+[0]*right
                    data.append(sent)
                line_control+=1
                if line_control==10000:
                    break
        read_file.close()
        if len(Lengths)/2 !=len(Y):
            print 'len(Lengths)/2 !=len(Y)'
            exit(0)
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            question=tokens[1].strip().lower().split() 
            answer=tokens[2].strip().lower().split()   
            if len(question)>max_truncate or len(answer)>max_truncate or len(question)< 2 or len(answer)<2:
                continue #skip this pair
            else:
                Y.append(int(tokens[0])) 
                sents=[question, answer]
                for i in [0,1]:
                    sent=[]
                    words=sents[i]
                    #true_lengths.append(len(words))
                    length=0
                    for word in words:
                        id=word2id.get(word)
                        if id is not None:
                            sent.append(id)
                            length+=1
                    if length==0:
                        print 'shit sentence: ', sents[i]
                        #exit(0)
                        break
                    Lengths.append(length)
                    left=(maxlength-length)/2
                    right=maxlength-left-length
                    leftPad.append(left)
                    rightPad.append(right)
     
                    sent=[0]*left+sent+[0]*right
                    data.append(sent)
                line_control+=1
                #if line_control==500:
                #    break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')
        #return T.cast(shared_y, 'int32') # for gpu
        #return shared_y


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
                                
    train_set_y=shared_dataset(trainY)                             
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1

def load_SICK_corpus(vocabFile, trainFile, testFile, max_truncate,maxlength, entailment): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            if entailment:
                Y.append(int(tokens[2])) 
            else:
                Y.append(float(tokens[3])) 
            #question
            for i in [0,1]:
                sent=[]
                words=tokens[i].strip().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==500:
            #    break
        read_file.close()
        if len(Lengths)/2 !=len(Y):
            print 'len(Lengths)/2 !=len(Y)'
            exit(0)
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            if entailment:
                Y.append(int(tokens[2])) 
            else:
                Y.append(float(tokens[3])) 
            #Y.append(int(tokens[0]))
            for i in [0,1]:
                sent=[]
                words=tokens[i].strip().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right) 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==200:
            #    break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        #return T.cast(shared_y, 'int64')
        return T.cast(shared_y, 'int64') # 
        #return shared_y
    def shared_dataset_float(data_y, borrow=True):
        return theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX), borrow=borrow)


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
    
    if entailment:                            
        train_set_y=shared_dataset(trainY)                             
        test_set_y = shared_dataset(testY)
    else:
        train_set_y=shared_dataset_float(trainY)                             
        test_set_y = shared_dataset_float(testY)        
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1

def load_SICK_corpus_binary_feature(vocabFile, trainFile, testFile, max_truncate,maxlength, entailment): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        binarys=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            if entailment:
                Y.append(int(tokens[2])) 
            else:
                Y.append(float(tokens[3])) 
            #question
            for i in [0,1]:
                sent=[]
                words=tokens[i].strip().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #binary feature
            words1=tokens[0].strip().split()  
            words2=tokens[1].strip().split()  
            set1=set(words1)
            set2=set(words2)
            len1=len(words1)
            len2=len(words2)
            binary1=[]
            binary2=[]
            for word in words1:
                if word in set2:
                    binary1.append(1.0)
                else:
                    binary1.append(1e-10)
            binary1=[0.0]*((maxlength-len1)/2)+binary1+[0.0]*(maxlength-(maxlength-len1)/2-len1)
            for word in words2:
                if word in set1:
                    binary2.append(1.0)
                else:
                    binary2.append(1e-10)
            binary2=[0.0]*((maxlength-len2)/2)+binary2+[0.0]*(maxlength-(maxlength-len2)/2-len2)            
            binarys.append(binary1)
            binarys.append(binary2)
            
            
            line_control+=1
            #if line_control==500:
            #    break
        read_file.close()
        if len(Lengths)/2 !=len(Y):
            print 'len(Lengths)/2 !=len(Y)'
            exit(0)
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(binarys), numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        binarys=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            if entailment:
                Y.append(int(tokens[2])) 
            else:
                Y.append(float(tokens[3])) 
            #Y.append(int(tokens[0]))
            for i in [0,1]:
                sent=[]
                words=tokens[i].strip().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right) 
                sent=[0]*left+sent+[0]*right
                data.append(sent)

            #binary feature
            words1=tokens[0].strip().split()  
            words2=tokens[1].strip().split()  
            set1=set(words1)
            set2=set(words2)
            len1=len(words1)
            len2=len(words2)
            binary1=[]
            binary2=[]
            for word in words1:
                if word in set2:
                    binary1.append(1.0)
                else:
                    binary1.append(1e-10)
            binary1=[0.0]*((maxlength-len1)/2)+binary1+[0.0]*(maxlength-(maxlength-len1)/2-len1)
            for word in words2:
                if word in set1:
                    binary2.append(1.0)
                else:
                    binary2.append(1e-10)
            binary2=[0.0]*((maxlength-len2)/2)+binary2+[0.0]*(maxlength-(maxlength-len2)/2-len2)            
            binarys.append(binary1)
            binarys.append(binary2)
            
            line_control+=1
            #if line_control==200:
            #    break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(binarys),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, binary_train, trainY, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, binary_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        #return T.cast(shared_y, 'int64')
        return T.cast(shared_y, 'int64') # 
        #return shared_y
    def shared_dataset_float(data_y, borrow=True):
        return theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX), borrow=borrow)


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
    
    if entailment:                            
        train_set_y=shared_dataset(trainY)                             
        test_set_y = shared_dataset(testY)
    else:
        train_set_y=shared_dataset_float(trainY)                             
        test_set_y = shared_dataset_float(testY)        
    
    train_binary=shared_dataset_float(binary_train)
    test_binary=shared_dataset_float(binary_test)

    rval = [(indices_train,train_binary, train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_binary, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1
def transfer_wordlist_2_idlist_with_maxlen(token_list, vocab_map, maxlen):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:
        id=vocab_map.get(word)
        if id is None: # if word was not in the vocabulary
            id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
            vocab_map[word]=id
        idlist.append(id)
    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list
def load_SNLI_dataset(maxlen=40):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"
    files=['train.txt', 'test.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[0])  # keep label be 0 or 1
                sentence_wordlist_l=parts[1].strip().split()
                sentence_wordlist_r=parts[2].strip().split()
                if len(sentence_wordlist_l) > max_sen_len:
                    max_sen_len=len(sentence_wordlist_l)
                if len(sentence_wordlist_r) > max_sen_len:
                    max_sen_len=len(sentence_wordlist_r)
                labels.append(label)
                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen)
                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels), 'pairs'
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len       
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id

def load_word2vec():
    word2vec = {}
    
    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    for line in f:    
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])
            
    print "==> word2vec is loaded"
    
    return word2vec 
