import numpy
from itertools import izip
from xml.sax.saxutils import escape
from nltk.tokenize import TreebankWordTokenizer
import collections
from nltk.corpus import wordnet as wn
from scipy import spatial

def extract_pairs(path, inputfile):
    read_file=open(path+inputfile, 'r')
    write_file=open(path+'dev.txt', 'w')
    line_no=0
    for line in read_file:
        line_no+=1
        if line_no==1:
            continue
        parts=line.strip().split('\t')
        if parts[0]=='neutral':
            write_file.write('0\t'+' '.join(TreebankWordTokenizer().tokenize(parts[5]))+'\t'+' '.join(TreebankWordTokenizer().tokenize(parts[6]))+'\n')
        elif parts[0]=='contradiction':
            write_file.write('1\t'+' '.join(TreebankWordTokenizer().tokenize(parts[5]))+'\t'+' '.join(TreebankWordTokenizer().tokenize(parts[6]))+'\n')
        elif parts[0]=='entailment':
            write_file.write('2\t'+' '.join(TreebankWordTokenizer().tokenize(parts[5]))+'\t'+' '.join(TreebankWordTokenizer().tokenize(parts[6]))+'\n')
    write_file.close()
    read_file.close()
    print line_no
        

def Extract_Vocab(path, train, dev, test):
    #consider all words, including unknown from word2vec, because some sentence 
    '''
    readFile=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    dim=300
    word2vec=set()
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            word2vec.add(tokens[0])
    readFile.close()
    print 'word2vec vocab loaded over...'    
    '''
    files=[train, dev, test]
    writeFile=open(path+'vocab_lower_in_word2vec_removed_overlap.txt', 'w')
    vocab={}
    count=0
    for file in files:
        readFile=open(path+file, 'r')
        for line in readFile:
            tokens=line.strip().split('\t')
            for i in [1,2]:
                words=tokens[i].strip().lower().split()
                for word in words:
                    key=vocab.get(word)
                    if key is None:
                        count+=1
                        vocab[word]=count
                        writeFile.write(str(count)+'\t'+word+'\n')
                        
        readFile.close()
    writeFile.close()
    print 'total words: ', count

def transcate_word2vec_into_entailment_vocab(rootPath):
    readFile=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    dim=300
    word2vec={}
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            word2vec[tokens[0]]=map(float, tokens[1:])
    readFile.close()
    print 'word2vec loaded over...'
    readFile=open(rootPath+'vocab_lower_in_word2vec_removed_overlap.txt', 'r')
    writeFile=open(rootPath+'vocab_lower_in_word2vec_embs_300d_removed_overlap.txt', 'w')
    random_emb=list(numpy.random.uniform(-0.01,0.01,dim))
    for line in readFile:
        tokens=line.strip().split()
        emb=word2vec.get(tokens[1])
        if emb is None:
            emb=random_emb
        writeFile.write(tokens[1]+'\t'+' '.join(map(str, emb))+'\n')
    writeFile.close()
    readFile.close()
    print 'word2vec trancate over'    

def compute_map_mrr(file, probs):
    #file
    testread=open(file, 'r')
    separate=[]
    labels=[]
    pre_q=' '
    line_no=0
    for line in testread:
        parts=line.strip().split('\t')
        if parts[0]!=pre_q:
            separate.append(line_no)
        labels.append(int(parts[2]))
        pre_q=parts[0]
        line_no+=1
    testread.close()
    separate.append(line_no)#the end of file
    #compute MAP, MRR
    question_no=len(separate)-1
    all_map=0.0
    all_mrr=0.0
    all_corr_answer=0
    for i in range(question_no):
        sub_labels=labels[separate[i]:separate[i+1]]
        sub_probs=probs[separate[i]:separate[i+1]]
        sub_dict = [(prob, label) for prob, label in izip(sub_probs, sub_labels)] # a list of tuple
        #sorted_probs=sorted(sub_probs, reverse = True)
        sorted_tuples=sorted(sub_dict,key=lambda tup: tup[0], reverse = True) 
        map=0.0
        find=False
        corr_no=0
        #MAP
        for index, (prob,label) in enumerate(sorted_tuples):
            if label==1:
                corr_no+=1 # the no of correct answers
                all_corr_answer+=1
                map+=1.0*corr_no/(index+1)
                find=True
        #MRR
        for index, (prob,label) in enumerate(sorted_tuples):
            if label==1:
                all_mrr+=1.0/(index+1)
                break # only consider the first correct answer              
        if find is False:
            print 'Did not find correct answers'
            exit(0)
        map=map/corr_no
        all_map+=map
    MAP=all_map/question_no
    MRR=all_mrr/question_no

    
    return MAP, MRR
    '''
    #compute MAP, MRR
    question_no=len(separate)-1
    all_map=0.0
    all_mrr=0.0
    all_corr_answer=0
    for i in range(question_no):
        sub_labels=labels[separate[i]:separate[i+1]]
        sub_probs=probs[separate[i]:separate[i+1]]
        all_map+=average_precision_score(numpy.array(sub_labels), numpy.array(sub_probs))  
        sub_dict = {k: v for k, v in izip(sub_probs, sub_labels)}
        sorted_probs=sorted(sub_probs, reverse = True) 

        find=False

        for index, prob in enumerate(sorted_probs):
            if sub_dict[prob]==1:
                all_corr_answer+=1
                all_mrr+=1.0/(index+1)
                find=True
                
        if find is False:
            print 'Did not find correct answers'
            exit(0)
    MAP=all_map/question_no
    MRR=all_mrr/all_corr_answer
    '''
              
def reform_for_bleu_nist(trainFile):#not useful
    #first src file
    read_train=open(trainFile, 'r')
    write_src=open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/train_src.xml', 'w')
    write_src.write('<mteval>'+'\n'+'<srcset setid="WMT08" srclang="Czech">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_src.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[0])+' </seg>\n</p>\n')
        id+=1
    write_src.write('</doc>\n</srcset>\n</mteval>\n')
    write_src.close()
    read_train.close()
    #second, ref
    read_train=open(trainFile, 'r')
    write_ref=open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/train_ref.xml', 'w')
    write_ref.write('<mteval>'+'\n'+'<refset setid="WMT08" srclang="Czech" trglang="English" refid="reference01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_ref.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[0])+' </seg>\n</p>\n')
        id+=1
    write_ref.write('</doc>\n</refset>\n</mteval>\n')
    write_ref.close()
    read_train.close()     
    #third, sys
    read_train=open(trainFile, 'r')
    write_sys=open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/train_sys.xml', 'w')
    write_sys.write('<mteval>'+'\n'+'<tstset setid="WMT08" srclang="Czech" trglang="English" sysid="system01">'+'\n'+'<doc docid="train" genre="nw">'+'\n')
    id=1
    for line in read_train:
        parts=line.strip().split('\t')
        write_sys.write('<p>'+'\n'+'<seg id="'+str(id)+'">'+' '+escape(parts[1])+' </seg>\n</p>\n')
        id+=1
    write_sys.write('</doc>\n</tstset>\n</mteval>\n')
    write_sys.close()
    read_train.close()        
        
def putAllMtTogether():
    pathroot='/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST'
    train_files=[#pathroot+'/result_train/BLEU1-seg.scr',
                 #pathroot+'/result_train/BLEU2-seg.scr',pathroot+'/result_train/BLEU3-seg.scr',
                 pathroot+'/result_train/BLEU4-seg.scr',
                 #pathroot+'/result_train/NIST1-seg.scr',
                 #pathroot+'/result_train/NIST2-seg.scr',
                 #pathroot+'/result_train/NIST3-seg.scr',pathroot+'/result_train/NIST4-seg.scr',
                 pathroot+'/result_train/NIST5-seg.scr'
                 ]
    
    test_files=[#pathroot+'/result_test/BLEU1-seg.scr',
                #pathroot+'/result_test/BLEU2-seg.scr',pathroot+'/result_test/BLEU3-seg.scr',
                pathroot+'/result_test/BLEU4-seg.scr',
                 #pathroot+'/result_test/NIST1-seg.scr',
                 #pathroot+'/result_test/NIST2-seg.scr',
                 #pathroot+'/result_test/NIST3-seg.scr',pathroot+'/result_test/NIST4-seg.scr',
                 pathroot+'/result_test/NIST5-seg.scr',
                  #pathroot+'maxsim-v1.01/paraphrase/test.score'
                  ]

    #posi=[4, 4,4,4,4,  4,4,4,4]
    posi=[4, 4]
    size=len(posi)
    
    train_write=open(pathroot+'/result_train/concate_2mt_train.txt', 'w')
    scores=[]
    for i in range(size):
        read_file=open(train_files[i], 'r')
        list_values=[]
        for line in read_file:
            tokens=line.strip().split()
            list_values.append(tokens[posi[i]])
        read_file.close()
        scores.append(list_values)
    values_matrix=numpy.array(scores)
    col=values_matrix.shape[1]
    for j in range(col):
        for i in range(size):
            train_write.write(values_matrix[i,j]+'\t')
        train_write.write('\n')
    train_write.close()
    #test
    test_write=open(pathroot+'/result_test/concate_2mt_test.txt', 'w')
    scores=[]
    for i in range(size):
        read_file=open(test_files[i], 'r')
        list_values=[]
        for line in read_file:
            tokens=line.strip().split()
            list_values.append(tokens[posi[i]])
        read_file.close()
        scores.append(list_values)
    values_matrix=numpy.array(scores)
    col=values_matrix.shape[1]
    for j in range(col):
        for i in range(size):
            test_write.write(values_matrix[i,j]+'\t')
        test_write.write('\n')
    test_write.close()
    print 'finished'            

def two_word_matching_methods(path, trainfile, testfile):
    stop_word_list=open(path+'short-stopwords.txt', 'r')
    stop_words=set()
    
    for line in stop_word_list:
        word=line.strip()
        stop_words.add(word)
    stop_word_list.close()
    print 'totally ', len(stop_words), ' stop words'
    #word 2 idf
    word2idf={}
    for file in [trainfile, testfile]:
        read_file=open(path+file, 'r')
        for line in read_file:
            parts=line.strip().split('\t')
            for i in [0,1]:
                sent2set=set(parts[i].split())# do not consider repetition
                for word in sent2set:
                    if word not in stop_words:
                        count=word2idf.get(word,0)
                        word2idf[word]=count+1
        read_file.close()
        
    '''   
    #train file
    read_train=open(path+trainfile, 'r')
    write_train=open(path+'train_word_matching_scores.txt','w')
    for line in read_train:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[0].split()
        answer=parts[1].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        write_train.write(str(WC)+' '+str(WWC)+'\n')
    write_train.close()
    read_train.close()
    
    #test file
    read_test=open(path+testfile, 'r')
    write_test=open(path+'test_word_matching_scores.txt','w')
    for line in read_test:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[0].split()
        answer=parts[1].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        write_test.write(str(WC)+' '+str(WWC)+'\n')
    write_test.close()
    read_test.close()             
    print 'two word matching values generated' 
    '''
    WC_train=[]
    WWC_train=[]
    #train file
    read_train=open(path+trainfile, 'r')
    #write_train=open(path+'train_word_matching_scores_normalized.txt','w')
    for line in read_train:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[0].split()
        answer=parts[1].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        WC_train.append(WC)
        WWC_train.append(WWC)
        #write_train.write(str(WC)+' '+str(WWC)+'\n')
    #write_train.close()
    read_train.close()
    
    #test file
    WC_test=[]
    WWC_test=[]
    read_test=open(path+testfile, 'r')
    #write_test=open(path+'test_word_matching_scores.txt','w')
    for line in read_test:
        parts=line.strip().split('\t')
        WC=0
        WWC=0
        question=parts[0].split()
        answer=parts[1].split()
        for word in question:
            if word not in stop_words and word in answer:
                WC+=1
                WWC+=1.0/word2idf.get(word)
        WC_test.append(WC)
        WWC_test.append(WWC)
        #write_test.write(str(WC)+' '+str(WWC)+'\n')
    #write_test.close()
    read_test.close()   
    WC_overall=WC_train+WC_test
    max_WC=numpy.max(WC_overall)          
    min_WC=numpy.min(WC_overall)
    
    write_train=open(path+'train_word_matching_scores_normalized.txt','w')
    for index,wc in enumerate(WC_train):
        wc=(wc-min_WC)*1.0/(max_WC-min_WC)
        write_train.write(str(wc)+' '+str(WWC_train[index])+'\n')
    write_train.close()
    write_test=open(path+'test_word_matching_scores_normalized.txt','w')
    for index,wc in enumerate(WC_test):
        wc=(wc-min_WC)*1.0/(max_WC-min_WC)
        write_test.write(str(wc)+' '+str(WWC_test[index])+'\n')
    write_test.close()    
    
    print 'two word matching values generated' 
    
def remove_overlap_words(path, inputfile, title):
    readfile=open(path+inputfile, 'r')
    writefile=open(path+title+'_removed_overlap.txt', 'w')
    for line in readfile:
        parts=line.strip().lower().split('\t')
        overlap=set(parts[1].split()).intersection(set(parts[2].split()))
        sent1=' '
        for i in parts[1].split():
            if i not in overlap:
                sent1+=i+' '
        sent2=' '
        for i in parts[2].split():
            if i not in overlap:
                sent2+=i+' '  
        if len(sent1.strip())==0:
            sent1='<empty>'
        if len(sent2.strip())==0:
            sent2='<empty>'
        
        writefile.write(parts[0]+'\t'+sent1.strip()+'\t'+sent2.strip()+'\n')
    readfile.close()
    writefile.close()        
def extract_synonyms_for_token(token):
    synonyms_dog=[]
    for synset in wn.synsets(token):
        synonyms_dog+=synset.lemma_names()            
    return set(synonyms_dog)        

def extract_hypernyms_for_token(token):
    results=[]
    synsetss=wn.synsets(token)
    if len(synsetss)>0:
        apple = synsetss[0]
        hyperapple = set([i for i in apple.closure(lambda s:s.hypernyms())])
        for word in hyperapple:
            results+=word.lemma_names()
    return results                   

def syn_relation(token1, token2):
    syn1=extract_synonyms_for_token(token1)
    syn2=extract_synonyms_for_token(token2)
    #print syn1
    #print syn2
    if token1 in syn2 or token2 in syn1:
        return True
    else:
        return False
def hyper_relation(token1, token2):
    hyper1=extract_hypernyms_for_token(token1)
    if token2 in hyper1:
        return True
    else:
        return False
def extract_synonyms_hypernyms_antonyms(path, trainfile, testfile):
    readtrain=open(path+trainfile, 'r')
    readtest=open(path+testfile, 'r')
    writesyn=open(path+'synonyms.txt', 'w')
    writehyper=open(path+'hypernyms.txt', 'w')
    writeanto=open(path+'antonyms.txt', 'w')
    pair_train=set()
    pair_train_entail=set()
    pair_train_nonentail=set()
    pair_train_freq=collections.defaultdict(int)
    pair_test=set()
    syn_set=set()
    hyper_set=set()
    for line in readtrain:
        parts=line.split('\t')
        sent1=parts[1].strip().split()
        sent2=parts[2].strip().split()
        if len(sent1) >0 and len(sent2) >0:
            for token1 in sent1:
                for token2 in sent2:
                    pair_train.add((token1, token2))
                    if parts[0]=='2': # entailment
                        pair_train_entail.add((token1, token2))
                    else:
                        pair_train_nonentail.add((token1, token2))
                    pair_train_freq[(token1,token2)]+=1
    readtrain.close()
    for line in readtest:
        parts=line.split('\t')
        sent1=parts[1].strip().split()
        sent2=parts[2].strip().split()
        if len(sent1) >0 and len(sent2) >0:
            for token1 in sent1:
                for token2 in sent2:
                    pair_test.add((token1, token2))
    readtest.close()
    #filter
    both_set=pair_train | pair_test
    for (token1,token2) in both_set:
        if syn_relation(token1, token2):
            syn_set.add((token1, token2))
            writesyn.write(token1+'\t'+token2+'\n')
        if hyper_relation(token1, token2):
            hyper_set.add((token1, token2))
            writehyper.write(token1+'\t'+token2+'\n')
    writesyn.close()
    writehyper.close()
    readFile=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    dim=300
    word2vec={}
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            word2vec[tokens[0]]=map(float, tokens[1:])
    readFile.close()
    print 'word2vec loaded over...'
    possible_anto=pair_train_nonentail-pair_train_entail-syn_set-hyper_set
    for (token1,token2) in possible_anto:
        if pair_train_freq.get((token1,token2))>=2:
            emb1=word2vec.get(token1)
            emb2=word2vec.get(token2)
            if emb1 is not None and emb2 is not None:
                simi=1 - spatial.distance.cosine(emb1, emb2)
                if simi>0.4:
                    writeanto.write(token1+'\t'+token2+'\n')
    writeanto.close()
    print 'over'
def features_for_nonoverlap_pairs(path, inputfile, title):
    
#     readFile=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
#     dim=300
#     word2vec={}
#     for line in readFile:
#         tokens=line.strip().split()
#         if len(tokens)<dim+1:
#             continue
#         else:
#             word2vec[tokens[0]]=map(float, tokens[1:])
#     readFile.close()
#     print 'word2vec loaded over...'

    syn_set=set()
    hyper_set=set()    
    anto_set=set()
    read_syn=open(path+'synonyms.txt', 'r')
    for line in read_syn:
        syn_set.add((line.strip().split()[0], line.strip().split()[1]))
    read_syn.close()
    read_hyper=open(path+'hypernyms.txt', 'r')
    for line in read_hyper:
        hyper_set.add((line.strip().split()[0], line.strip().split()[1]))
    read_hyper.close()
    read_anto=open(path+'antonyms.txt', 'r')
    for line in read_anto:
        anto_set.add((line.strip().split()[0], line.strip().split()[1]))
    read_anto.close()
            
    readfile=open(path+inputfile, 'r')
    writefile=open(path+title+'_rule_features_negation_syn_hyper1_hyper2.txt', 'w')
    for line in readfile:
        parts=line.split('\t')
        
        negation=0.0
        sent_concate=parts[1].split()+parts[2].split()
        #if ('no' in sent_concate or 'not' in sent_concate or 'nobody' in sent_concate or "isn't" in sent_concate) and (len(parts[0].strip().split()) < 5 and  len(parts[1].strip().split()) < 5  ):
        if ('no' in sent_concate or 'not' in sent_concate or 'nobody' in sent_concate or "isn't" in sent_concate):
            negation=1.0
        digit=0.0
        if ('one' in sent_concate or 'two' in sent_concate or 'three' in sent_concate or 'four' in sent_concate or 'five' in sent_concate):
            digit=1.0
        len1_w=len(parts[1].strip().split())
        len2_w=len(parts[2].strip().split())
#         len1_c=len(parts[1].strip())
#         len2_c=len(parts[2].strip())
        #synonym and hypernym, antonyms

        num_syn=0.0
        num_hyp1=0.0
        num_hyp2=0.0
        num_ant=0.0
        sent1=parts[1].strip().split()
        sent2=parts[2].strip().split()
        if len(sent1)>0 and len(sent2)>0:
            for token1 in sent1:
                for token2 in sent2:
                    if (token1, token2) in syn_set or (token2, token1) in syn_set:
                        num_syn+=1
                    if (token1,token2) in hyper_set:
                        num_hyp1+=1
                    if (token2,token1) in hyper_set:
                        num_hyp2+=1
                    if (token1,token2) in anto_set or (token2,token1) in anto_set:
                        num_ant+=1
                
        writefile.write(str(negation)+'\t'+str(num_syn)+'\t'+str(num_hyp1)+'\t'+str(num_hyp2)+'\n')
    writefile.close()
    readfile.close()
if __name__ == '__main__':
    path='/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/'
    #extract_pairs(path, 'snli_1.0_dev.txt')
    #Extract_Vocab(path, 'train_removed_overlap.txt', 'dev_removed_overlap.txt', 'test_removed_overlap.txt')
    #transcate_word2vec_into_entailment_vocab(path)
    #compute_map_mrr(path+'test_filtered.txt')
    #reform_for_bleu_nist(path+'WikiQA-train.txt')
    #putAllMtTogether()
    #two_word_matching_methods(path, 'WikiQA-train.txt', 'test_filtered.txt')
    #remove_overlap_words(path, 'test.txt', 'test')
    #extract_synonyms_hypernyms_antonyms(path, 'train_removed_overlap.txt', 'test_removed_overlap.txt')
    features_for_nonoverlap_pairs(path, 'train_removed_overlap.txt', 'train')
    


