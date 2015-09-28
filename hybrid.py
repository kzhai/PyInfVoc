import sys
import re;
import time;
import string;
import random;
import math;

import numpy;
import scipy;
import scipy.special;
import scipy.io;
import scipy.sparse;
#import nchar;

import nltk;
import nltk.corpus;

"""
Implements online variational Bayesian for LDA.
"""
class Hybrid():
    """
    """
    def __init__(self,
                 minimum_word_length=3,
                 maximum_word_length=20,
                 dict_list=None,
                 N=3,
                 word_model_smooth=1e6,
                 char_list=string.lowercase,
                 #char_list=string.lowercase + string.digits
                 ranking_statistics_scale=1e3
                ):
        from nltk.stem.porter import PorterStemmer
        self._stemmer = PorterStemmer();

        self._minimum_word_length = minimum_word_length;
        self._maximum_word_length = maximum_word_length;
        
        '''
        self._word_model_smooth = word_model_smooth;
        self._char_list = char_list;
        self._n_char_model = N;

        if dict_list != None:
            tokens = [];
            for line in open(dict_list, 'r'):
                line = line.strip();
                if len(line) <= 0:
                    continue;
                #tokens.append(line);
                tokens.append(self._stemmer.stem(line));
            #tokens = set(tokens);
            #self._word_model = nchar.NcharModel(self._n_char_model, tokens, self._word_model_smooth, self._maximum_word_length, self._minimum_word_length, self._char_list);
        else:
            self._word_model = None;
        '''
        self._word_model = None;
        
        self._ranking_statistics_scale = ranking_statistics_scale;
        
    """
    """
    def _initialize(self,
                    vocab,
                    number_of_topics,
                    number_of_documents,
                    batch_size,
                    expected_truncation_size,
                    alpha_theta=1e-2,
                    alpha_beta=1e6,
                    tau=1.,
                    kappa=0.5,
                    refine_vocab_interval=10,
                    save_word_trace=False,
                    ranking_smooth_factor=1e-12,
                    #gamma_converge_threshold=1e-3,
                    #number_of_samples=50,
                    number_of_samples=10,
                    #burn_in_sweeps=2
                    burn_in_sweeps=5
                    ):
        self._number_of_topics = number_of_topics;
        self._number_of_documents = number_of_documents;
        self._batch_size = batch_size;
        
        self._word_to_index = {};
        self._index_to_word = {};
        for word in set(vocab):
            self._index_to_word[len(self._index_to_word)] = word;
            self._word_to_index[word] = len(self._word_to_index);
        vocab = self._index_to_word.keys();
        
        self._new_words = [len(self._index_to_word)];
        
        self._word_trace=None;
        if save_word_trace:
            self._word_trace = [];
            for index in vocab:
                self._word_trace.append(numpy.zeros((self._number_of_topics, self._number_of_documents/self._batch_size + 1), dtype='int32') + numpy.iinfo(numpy.int32).max);
            
        self._index_to_nupos = [];
        self._nupos_to_index = [];
        for k in xrange(self._number_of_topics):
            self._index_to_nupos.append(dict());
            self._nupos_to_index.append(dict());
            
            random.shuffle(vocab);
            
            for index in vocab:
                self._nupos_to_index[k][len(self._nupos_to_index[k])] = index;
                self._index_to_nupos[k][index] = len(self._index_to_nupos[k]);
        
        self._alpha_theta = alpha_theta;
        self._alpha_beta = alpha_beta;
        self._tau = tau;
        self._kappa = kappa;
        
        self._truncation_size = [];
        self._truncation_size_prime = [];
        self._nu_1 = {};
        self._nu_2 = {};
        for k in xrange(self._number_of_topics):
            self._truncation_size.append(len(self._index_to_nupos[k]));
            self._truncation_size_prime.append(len(self._index_to_nupos[k]));
            self._nu_1[k] = numpy.ones((1, self._truncation_size[k]));
            self._nu_2[k] = numpy.ones((1, self._truncation_size[k])) * self._alpha_beta;

        self._expected_truncation_size = expected_truncation_size;
        
        #self._gamma_converge_threshold = gamma_converge_threshold;
        self._number_of_samples = number_of_samples;
        self._burn_in_sweeps = burn_in_sweeps;
        assert(self._burn_in_sweeps < self._number_of_samples);
        
        self._ranking_smooth_factor = ranking_smooth_factor;
        self._reorder_vocab_interval = refine_vocab_interval;
        
        self._counter = 0;

        self._ranking_statistics = [];
        for k in xrange(self._number_of_topics):
            self._ranking_statistics.append(nltk.probability.FreqDist());

            for index in self._index_to_nupos[k]:
                #self._ranking_statistics[k].inc(index, self._ranking_smooth_factor);
                self._ranking_statistics[k][index] += self._ranking_smooth_factor;

                '''
                if self._word_model != None:
                    self._ranking_statistics[k].inc(index, self._word_model.probability(self._index_to_word[index]) * self._ranking_statistics_scale);
                else:
                    self._ranking_statistics[k].inc(index, self._ranking_smooth_factor);
                '''

        self._document_topic_distribution = None;
        
        if self._word_trace!=None:
            self.update_word_trace();

    def update_word_trace(self):
        if self._counter>self._number_of_documents/self._batch_size:
            return;
        for topic_index in xrange(self._number_of_topics):
            temp_keys = self._ranking_statistics[topic_index].keys();
            for word_rank in xrange(len(temp_keys)):
                self._word_trace[temp_keys[word_rank]][topic_index, self._counter:] = word_rank+1;
                
    def parse_doc_list(self, docs):
        if (type(docs).__name__ == 'str'):
            temp = list()
            temp.append(docs)
            docs = temp
    
        assert self._batch_size == len(docs);

        batch_documents = [];
        
        for d in xrange(self._batch_size):
            '''
            docs[d] = docs[d].lower();
            docs[d] = re.sub(r'-', ' ', docs[d]);
            docs[d] = re.sub(r'[^a-z ]', '', docs[d]);
            docs[d] = re.sub(r'[^a-z0-9 ]', '', docs[d]);
            docs[d] = re.sub(r' +', ' ', docs[d]);
            
            words = [];
            for word in docs[d].split():
                if word in nltk.corpus.stopwords.words('english'):
                    continue;
                word = self._stemmer.stem(word);
                if word in nltk.corpus.stopwords.words('english'):
                    continue;
                if len(word)>=self.maximum_word_length or len(word)<=self._minimum_word_length
                    continue;
                words.append(word);
            '''
            
            words = [word for word in docs[d].split() if len(word)<=self._maximum_word_length and len(word)>=self._minimum_word_length];
            
            document_topics = numpy.zeros((self._number_of_topics, len(words)));

            for word_index in xrange(len(words)):
                word = words[word_index];
                # valid only if limiting the ranking statistics 
                if word not in self._word_to_index:
                    #if this word never appeared before
                    index = len(self._word_to_index);

                    self._index_to_word[len(self._index_to_word)] = word;
                    self._word_to_index[word] = len(self._word_to_index);
                    
                    if self._word_trace!=None:                    
                        self._word_trace.append(numpy.zeros((self._number_of_topics, self._number_of_documents/self._batch_size + 1), dtype='int32') + numpy.iinfo(numpy.int32).max);
                        
                    for topic in xrange(self._number_of_topics):
                        #self._ranking_statistics[topic].inc(index, self._ranking_smooth_factor);
                        self._ranking_statistics[topic][index] += self._ranking_smooth_factor;

                else:
                    index = self._word_to_index[word];
                        
                for topic in xrange(self._number_of_topics):
                    if index not in self._index_to_nupos[topic]:
                        # if this word is not in current vocabulary
                        self._nupos_to_index[topic][len(self._nupos_to_index[topic])] = index;
                        self._index_to_nupos[topic][index] = len(self._index_to_nupos[topic]);
                        
                        self._truncation_size_prime[topic] += 1;
                        
                    document_topics[topic, word_index]=self._index_to_nupos[topic][index];
                
            batch_documents.append(document_topics);
            
        if self._word_trace!=None:
            self.update_word_trace();
        
        self._new_words.append(len(self._word_to_index));

        return batch_documents;

    """
    Compute the aggregate digamma values, for phi update.
    """
    def compute_exp_weights(self):
        exp_weights = {};
        exp_oov_weights = {};
        
        for k in xrange(self._number_of_topics):
            psi_nu_1_k = scipy.special.psi(self._nu_1[k]);
            psi_nu_2_k = scipy.special.psi(self._nu_2[k]);
            psi_nu_all_k = scipy.special.psi(self._nu_1[k] + self._nu_2[k]);
            
            aggregate_psi_nu_2_minus_psi_nu_all_k = numpy.cumsum(psi_nu_2_k - psi_nu_all_k, axis=1);
            exp_oov_weights[k] = numpy.exp(aggregate_psi_nu_2_minus_psi_nu_all_k[0, -1]);
            
            aggregate_psi_nu_2_minus_psi_nu_all_k = numpy.hstack((numpy.zeros((1, 1)), aggregate_psi_nu_2_minus_psi_nu_all_k[:, :-1]));
            assert(aggregate_psi_nu_2_minus_psi_nu_all_k.shape==psi_nu_1_k.shape);
            
            exp_weights[k] = numpy.exp(psi_nu_1_k - psi_nu_all_k + aggregate_psi_nu_2_minus_psi_nu_all_k);

        return exp_weights, exp_oov_weights;
    
    """
    """
    def e_step(self, wordids, directory=None):
        batch_size = len(wordids);
        
        sufficient_statistics = {};
        for k in xrange(self._number_of_topics):
            sufficient_statistics[k] = numpy.zeros((1, self._truncation_size_prime[k]));
            
        batch_document_topic_distribution = numpy.zeros((batch_size, self._number_of_topics));
        #batch_document_topic_distribution = scipy.sparse.dok_matrix((batch_size, self._number_of_topics), dtype='int16');

        #log_likelihood = 0;
        exp_weights, exp_oov_weights = self.compute_exp_weights();
        
        # Now, for each document document_index update that document's phi_d for every words
        for document_index in xrange(batch_size):
            phi = numpy.random.random(wordids[document_index].shape);
            phi = phi / numpy.sum(phi, axis=0)[numpy.newaxis, :];
            phi_sum = numpy.sum(phi, axis=1)[:, numpy.newaxis];
            #assert(phi_sum.shape == (self.number_of_topics, 1));

            for sample_index in xrange(self._number_of_samples):
                for word_index in xrange(wordids[document_index].shape[1]):
                    phi_sum -= phi[:, word_index][:, numpy.newaxis];
                    # this is to get rid of the underflow error from the above summation, ideally, phi will become all integers after few iterations
                    phi_sum *= phi_sum > 0;
                    #assert(numpy.all(phi_sum >= 0));

                    temp_phi = phi_sum + self._alpha_theta;
                    #assert(temp_phi.shape == (self.number_of_topics, 1));
                    
                    for k in xrange(self._number_of_topics):
                        id = wordids[document_index][k, word_index];

                        if id >= self._truncation_size[k]:
                            # if this word is an out-of-vocabulary term
                            temp_phi[k, 0] *= exp_oov_weights[k];
                        else:
                            # if this word is inside current vocabulary
                            temp_phi[k, 0] *= exp_weights[k][0, id];

                    temp_phi /= numpy.sum(temp_phi);
                    #assert(temp_phi.shape == (self.number_of_topics, 1));

                    # sample a topic for this word
                    temp_phi = temp_phi.T[0];
                    temp_phi = numpy.random.multinomial(1, temp_phi)[:, numpy.newaxis];
                    #assert(temp_phi.shape == (self.number_of_topics, 1));
                    
                    phi[:, word_index][:, numpy.newaxis] = temp_phi;
                    phi_sum += temp_phi;
                    #assert(numpy.all(phi_sum >= 0));

                    # discard the first few burn-in sweeps
                    if sample_index >= self._burn_in_sweeps:
                        for k in xrange(self._number_of_topics):
                            id = wordids[document_index][k, word_index];
                            sufficient_statistics[k][0, id] += temp_phi[k, 0];

            batch_document_topic_distribution[document_index, :] = self._alpha_theta + phi_sum.T[0, :];
                        
        for k in xrange(self._number_of_topics):
            sufficient_statistics[k] /= (self._number_of_samples - self._burn_in_sweeps);

        return sufficient_statistics, batch_document_topic_distribution;

    """
    """
    def m_step(self, batch_size, sufficient_statistics, close_form_updates=False):
        #sufficient_statistics = self.sort_sufficient_statistics(sufficient_statistics);
        reverse_cumulated_phi = {};
        for k in xrange(self._number_of_topics):
            reverse_cumulated_phi[k] = self.reverse_cumulative_sum_matrix_over_axis(sufficient_statistics[k], 1);
        
        if close_form_updates:
            self._nu_1 = 1 + sufficient_statistics;
            self._nu_2 = self._alpha_beta + reverse_cumulated_phi;
        else:
            # Epsilon will be between 0 and 1, and says how much to weight the information we got from this mini-batch.
            self._epsilon = pow(self._tau + self._counter, -self._kappa);
            
            self.update_accumulate_sufficient_statistics(sufficient_statistics);

            for k in xrange(self._number_of_topics):
                if self._truncation_size[k] < self._truncation_size_prime[k]:
                    self._nu_1[k] = numpy.append(self._nu_1[k], numpy.ones((1, self._truncation_size_prime[k] - self._truncation_size[k])), 1);
                    self._nu_2[k] = numpy.append(self._nu_2[k], numpy.ones((1, self._truncation_size_prime[k] - self._truncation_size[k])), 1);
                    
                    self._truncation_size[k] = self._truncation_size_prime[k];
                    
                self._nu_1[k] += self._epsilon * (self._number_of_documents / batch_size * sufficient_statistics[k] + 1 - self._nu_1[k]);
                self._nu_2[k] += self._epsilon * (self._alpha_beta + self._number_of_documents / batch_size * reverse_cumulated_phi[k] - self._nu_2[k]);

    """
    """
    def update_accumulate_sufficient_statistics(self, sufficient_statistics):
        for k in xrange(self._number_of_topics):
            for index in self._index_to_word:
                #self._ranking_statistics[k].inc(index, -self._epsilon*self._ranking_statistics[k][index]);
                self._ranking_statistics[k][index] += -self._epsilon*self._ranking_statistics[k][index];
            for index in self._index_to_nupos[k]:
                if self._word_model != None:
                    adjustment = self._word_model.probability(self._index_to_word[index]) * self._ranking_statistics_scale;
                else:
                    adjustment = 1.;
                #self._ranking_statistics[k].inc(index, self._epsilon*adjustment*sufficient_statistics[k][0, self._index_to_nupos[k][index]]);
                self._ranking_statistics[k][index] += self._epsilon*adjustment*sufficient_statistics[k][0, self._index_to_nupos[k][index]];

    """
    """
    def prune_vocabulary(self):
        # Re-order the nu values
        new_index_to_nupos = [];
        new_nupos_to_index = [];
        new_nu_1 = {};
        new_nu_2 = {};
        for k in xrange(self._number_of_topics):
            if len(self._index_to_nupos[k]) < self._expected_truncation_size:
                new_nu_1[k] = numpy.zeros((1, len(self._index_to_nupos[k])));
                new_nu_2[k] = numpy.zeros((1, len(self._index_to_nupos[k])));
            else:
                new_nu_1[k] = numpy.zeros((1, self._expected_truncation_size));
                new_nu_2[k] = numpy.zeros((1, self._expected_truncation_size));
            new_index_to_nupos.append(dict());
            new_nupos_to_index.append(dict());
            for index in self._ranking_statistics[k].keys():
                if len(new_index_to_nupos[k])>=min(self._index_to_nupos[k], self._expected_truncation_size):
                    break;
                
                #if index in words_to_keep and index in self._index_to_nupos[k].keys():
                new_nupos_to_index[k][len(new_index_to_nupos[k])] = index;
                new_index_to_nupos[k][index] = len(new_index_to_nupos[k]);
                
                if index not in self._index_to_nupos[k]:
                    # TODO: this statement is never reached.
                    new_nu_1[k][0, new_index_to_nupos[k][index]] = 1;
                    new_nu_2[k][0, new_index_to_nupos[k][index]] = 1;
                else:
                    new_nu_1[k][0, new_index_to_nupos[k][index]] = self._nu_1[k][0, self._index_to_nupos[k][index]];
                    new_nu_2[k][0, new_index_to_nupos[k][index]] = self._nu_2[k][0, self._index_to_nupos[k][index]];

            self._truncation_size[k] = len(new_index_to_nupos[k]);
            self._truncation_size_prime[k] = self._truncation_size[k];
            
        self._index_to_nupos = new_index_to_nupos;
        self._nupos_to_index = new_nupos_to_index;
        self._nu_1 = new_nu_1;
        self._nu_2 = new_nu_2;

    """
    """
    def learning(self, batch):
        self._counter += 1;

        # This is to handle the case where someone just hands us a single document, not in a list.
        if (type(batch).__name__ == 'string'):
            temp = list();
            temp.append(batch);
            batch = temp;

        batch_size = len(batch);

        # Parse the document mini-batch
        clock = time.time();
        wordids = self.parse_doc_list(batch);
        clock_p_step = time.time() - clock;
        
        # E-step: hybrid approach, sample empirical topic assignment
        clock = time.time();
        sufficient_statistics, batch_document_topic_distribution = self.e_step(wordids);
        clock_e_step = time.time() - clock;
        
        # M-step: online variational inference
        clock = time.time();
        self.m_step(batch_size, sufficient_statistics);
        if self._counter % self._reorder_vocab_interval==0:
            self.prune_vocabulary();
        clock_m_step = time.time() - clock;
        
        print 'P-step, E-step and M-step take %d, %d, %d seconds respectively...' % (clock_p_step, clock_e_step, clock_m_step);
        
        return batch_document_topic_distribution;
    
    """
    """
    def reverse_cumulative_sum_matrix_over_axis(self, matrix, axis):
        cumulative_sum = numpy.zeros(matrix.shape);
        (k, n) = matrix.shape;
        if axis == 1:
            for j in xrange(n - 2, -1, -1):
                cumulative_sum[:, j] = cumulative_sum[:, j + 1] + matrix[:, j + 1];
        elif axis == 0:
            for i in xrange(k - 2, -1, -1):
                cumulative_sum[i, :] = cumulative_sum[i + 1, :] + matrix[i + 1, :];
    
        return cumulative_sum;

    def export_beta(self, exp_beta_path, top_display=-1):
        exp_weights, exp_oov_weights = self.compute_exp_weights();

        output = open(exp_beta_path, 'w');
        for k in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (k));
            
            i = 0;
            for type_index in reversed(numpy.argsort(exp_weights[k][0, :])):
                i += 1;
                output.write("%s\t%g\n" % (self._index_to_word[type_index], exp_weights[k][0, type_index]));
                if top_display > 0 and i >= top_display:
                    break;

        output.close();
