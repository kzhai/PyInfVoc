#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import optparse;

'''
from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

def retrieve_vocabulary(docs):
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    freq_dist = FreqDist();
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if word not in stopwords.words('english'):
                freq_dist.inc(word);

    return freq_dist
'''

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        corpus_name=None,
                        dictionary=None,
                        
                        # parameter set 2
                        truncation_level=1000,
                        number_of_topics=25,
                        number_of_documents=-1,

                        # parameter set 3
                        vocab_prune_interval=10,
                        snapshot_interval=10,
                        batch_size=-1,
                        training_iterations=-1,
                        
                        # parameter set 4
                        kappa=0.75,
                        tau=64.0,
                        alpha_theta=-1,
                        alpha_beta=1e3,
                        
                        # parameter set 5
                        #heldout_data=0.0,
                        enable_word_model=False
                        
                        #fix_vocabulary=False
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      help="the corpus name [None]")
    parser.add_option("--dictionary", type="string", dest="dictionary",
                      help="the dictionary file [None]")
    
    # parameter set 2
    parser.add_option("--truncation_level", type="int", dest="truncation_level",
                      help="desired truncation level [1000]");
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="second level truncation [25]");
    parser.add_option("--number_of_documents", type="int", dest="number_of_documents",
                      help="number of documents [-1]");

    # parameter set 3
    parser.add_option("--vocab_prune_interval", type="int", dest="vocab_prune_interval",
                      help="vocabuary rank interval [10]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [vocab_prune_interval]");
    parser.add_option("--batch_size", type="int", dest="batch_size",
                      help="batch size [-1 in batch mode]");
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="max iteration to run training [number_of_documents/batch_size]");
                      
    # parameter set 4
    parser.add_option("--kappa", type="float", dest="kappa",
                      help="learning rate [0.5]")
    parser.add_option("--tau", type="float", dest="tau",
                      help="learning inertia [64.0]")
    parser.add_option("--alpha_theta", type="float", dest="alpha_theta",
                      help="hyper-parameter for Dirichlet distribution of topics [1.0/number_of_topics]")
    parser.add_option("--alpha_beta", type="float", dest="alpha_beta",
                      help="hyper-parameter for Dirichlet process of distribution over words [1e3]")

    (options, args) = parser.parse_args();
    return options;

def main():
    #import option_parser;
    options = parse_args();

    # parameter set 2
    assert(options.number_of_documents>0);
    number_of_documents = options.number_of_documents;
    assert(options.number_of_topics>0);
    number_of_topics = options.number_of_topics;
    assert(options.truncation_level>0);
    truncation_level = options.truncation_level;

    # parameter set 3
    assert(options.vocab_prune_interval>0);
    vocab_prune_interval = options.vocab_prune_interval;
    snapshot_interval = vocab_prune_interval;
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    #assert(options.batch_size>0);
    batch_size = options.batch_size;
    #assert(number_of_documents % batch_size==0);
    training_iterations=number_of_documents/batch_size;
    if options.training_iterations>0:
        training_iterations=options.training_iterations;

    # parameter set 4
    assert(options.tau>=0);
    tau = options.tau;
    #assert(options.kappa>=0.5 and options.kappa<=1);
    assert(options.kappa>=0 and options.kappa<=1);
    kappa = options.kappa;
    if batch_size<=0:
        print "warning: running in batch mode..."
        kappa = 0;
    alpha_theta = 1.0/number_of_topics;
    if options.alpha_theta>0:
        alpha_theta=options.alpha_theta;
    assert(options.alpha_beta>0);
    alpha_beta = options.alpha_beta;
    
    # parameter set 5
    #heldout_data = options.heldout_data;

    # parameter set 1
    #assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    corpus_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);

    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%b%d-%H%M%S")+"";
    suffix += "-D%d" % (number_of_documents);
    suffix += "-K%d" % (number_of_topics)
    suffix += "-T%d" % (truncation_level);
    suffix += "-P%d" % (vocab_prune_interval);
    suffix += "-I%d" % (snapshot_interval);
    suffix += "-B%d" % (batch_size);
    suffix += "-O%d" % (training_iterations);
    suffix += "-t%d" % (tau);
    suffix += "-k%g" % (kappa);
    suffix += "-at%g" % (alpha_theta);
    suffix += "-ab%g" % (alpha_beta);
    suffix += "/";
    '''
    suffix += "-D%d-K%d-T%d-P%d-S%d-B%d-O%d-t%d-k%g-at%g-ab%g/" % (number_of_documents,
                                                                   number_of_topics,
                                                                   truncation_level,
                                                                   vocab_prune_interval,
                                                                   snapshot_interval,
                                                                   batch_size,
                                                                   training_iterations,
                                                                   tau,
                                                                   kappa,
                                                                   alpha_theta,
                                                                   alpha_beta);
    '''
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    
    dictionary_file = options.dictionary;
    if dictionary_file != None:
        dictionary_file = dictionary_file.strip();
        
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    options_output_file.write("dictionary_file=" + str(dictionary_file) + "\n");
    # parameter set 2
    options_output_file.write("number_of_documents=" + str(number_of_documents) + "\n");
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    options_output_file.write("truncation_level=" + str(truncation_level) + "\n");
    # parameter set 3
    options_output_file.write("vocab_prune_interval=" + str(vocab_prune_interval) + "\n");
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    options_output_file.write("batch_size=" + str(batch_size) + "\n");
    options_output_file.write("training_iterations=" + str(training_iterations) + "\n");
    # parameter set 4
    options_output_file.write("tau=" + str(tau) + "\n");
    options_output_file.write("kappa=" + str(kappa) + "\n");
    options_output_file.write("alpha_theta=" + str(alpha_theta) + "\n");
    options_output_file.write("alpha_beta=" + str(alpha_beta) + "\n");
    # parameter set 5    
    #options_output_file.write("heldout_data=" + str(heldout_data) + "\n");
    options_output_file.close()
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    print "dictionary_file=" + str(dictionary_file)
    # parameter set 2
    print "number_of_documents=" + str(number_of_documents)
    print "number_of_topics=" + str(number_of_topics)
    print "truncation_level=" + str(truncation_level)
    # parameter set 3
    print "vocab_prune_interval=" + str(vocab_prune_interval)
    print "snapshot_interval=" + str(snapshot_interval);
    print "batch_size=" + str(batch_size)
    print "training_iterations=" + str(training_iterations)
    # parameter set 4
    print "tau=" + str(tau)
    print "kappa=" + str(kappa)
    print "alpha_theta=" + str(alpha_theta)
    print "alpha_beta=" + str(alpha_beta)
    # parameter set 5
    #print "heldout_data=" + str(heldout_data)
    print "========== ========== ========== ========== =========="

    # Vocabulary
    #file = open(input_directory+'voc.dat', 'r');
    # Seed the vocabulary
    vocab = ['team'];

    # Documents
    train_docs = [];
    file = open(os.path.join(input_directory, 'train.dat'), 'r');
    for line in file:
        train_docs.append(line.strip());
    print "successfully load all training documents..."

    import hybrid;
    infvoc_inferencer = hybrid.Hybrid(3, 20)

    infvoc_inferencer._initialize(vocab,
                                  number_of_topics,
                                  number_of_documents,
                                  batch_size,
                                  truncation_level,
                                  alpha_theta,
                                  alpha_beta,
                                  tau,
                                  kappa,
                                  vocab_prune_interval,
                                  True
                                  );
    infvoc_inferencer.export_beta(os.path.join(output_directory, 'exp_beta-0'), 100);
    
    document_topic_distribution = None;

    # Run until we've seen number_of_documents documents. (Feel free to interrupt *much* sooner than this.)
    for iteration in xrange(training_iterations):
        if batch_size<=0:
            docset = train_docs;
        else:
            docset = train_docs[(batch_size * iteration) % len(train_docs) : (batch_size * (iteration+1) - 1) % len(train_docs) + 1];
            print "select documents from %d to %d" % ((batch_size * iteration) % (number_of_documents), (batch_size * (iteration+1) - 1) % number_of_documents + 1)

        clock = time.time();

        batch_gamma = infvoc_inferencer.learning(docset);
        
        if (infvoc_inferencer._counter % snapshot_interval == 0):
            infvoc_inferencer.export_beta(os.path.join(output_directory, 'exp_beta-' + str(infvoc_inferencer._counter)), 50);
            
        if document_topic_distribution is None:
            document_topic_distribution = batch_gamma;
        else:
            document_topic_distribution = numpy.vstack((document_topic_distribution, batch_gamma));
        
        clock = time.time()-clock;
        
        print "vocabulary size = %s" % (infvoc_inferencer._truncation_size);
        print 'training iteration %d finished in %f seconds: epsilon = %f' % (infvoc_inferencer._counter, clock, infvoc_inferencer._epsilon);

    gamma_path = os.path.join(output_directory, "gamma.txt");
    numpy.savetxt(gamma_path, document_topic_distribution);
    
    model_snapshot_path = os.path.join(output_directory, 'model-' + str(infvoc_inferencer._counter));
    cPickle.dump(infvoc_inferencer, open(model_snapshot_path, 'wb'));

if __name__ == '__main__':
    main()