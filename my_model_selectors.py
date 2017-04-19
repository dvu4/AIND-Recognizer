import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        
        N: number of data points (N = length of self.X)
        p: number of parameters ( p = n_components  : is number of states in HMM)
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        # list of BIC score
        list_scores = []
        
        # list of number states in HMM
        list_num_hidden_states = []
        
        N = len(self.X)
        
        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states)
                logL = model.score(self.X, self.lengths)
                
                p = N*N + 2*num_hidden_states*N - 1
                bic = -2 * logL + p * np.log(N)
                
                list_scores.append(bic)
                list_num_hidden_states.append(num_hidden_states)
                
            except:
                # eliminate non-viable models from consideration
                pass
        
        if list_scores:
            best_num_hidden_states = list_num_hidden_states[np.argmax(list_scores)] 
        else:
            best_num_hidden_states = self.n_constant
            
        #best_num_states = list_num_states[np.argmax(list_scores)] if list_scores else self.n_constant
        
        #print("[SelectorBIC] Result model n_compents: {}".format(best_num_hidden_states))
        
        return self.base_model(best_num_hidden_states)           
        #raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        # list of number states in HMM
        list_num_hidden_states = []
        
        list_logL = []
        
        sum_logL = 0
        
        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states)
                logL = model.score(self.X, self.lengths)
                sum_logL += logL
                
                list_logL.append(logL)
                list_num_hidden_states.append(num_hidden_states)
                
            except:
                # eliminate non-viable models from consideration
                pass
            
        M = len(list_num_hidden_states) # length of list of number of hidden states
        if M > 2:
            
            # list of DIC score
            list_scores = []
            
            for logL in list_logL:
                dic = logL - (sum_logL - logL)/(M - 1)
                list_scores.append(dic)
                
            best_num_hidden_states = list_num_hidden_states[np.argmax(list_scores)]
            
        elif M == 2:
            best_num_hidden_states = list_num_hidden_states[0]
        else:
            best_num_hidden_states = self.n_constant
            
        #print("[SelectorDIC] Result model n_compents: {}".format(best_num_hidden_states))
        return self.base_model(best_num_hidden_states) 
        #raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # n_components : is number of hidden states in HMM
        
        split_method = KFold()
        
        # list of mean score of each Cross-Validation
        list_scores = []
        
        # list of number states in HMM
        list_num_hidden_states = []
        
        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                if len(self.sequences) > 2: # Check if there are enough data to split
                    scores = []
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # training sequences
                        self.X, self.lengths =  combine_sequences(cv_train_idx, self.sequences)
                        # testing sequences
                        test_X, test_lengths =  combine_sequences(cv_test_idx, self.sequences)
            
                        model = self.base_model(num_hidden_states)
                        scores.append(model.score(test_X, test_lengths))
   
                    list_scores.append(np.mean(scores))
                
                else:
                    model = self.base_model(num_hidden_states)
                    list_scores.append(model.score(self.X, self.lengths))
                    
                list_num_hidden_states.append(num_hidden_states)
                
            except:
                # eliminate non-viable models from consideration
                pass
            
        if list_scores:
            best_num_hidden_states = list_num_hidden_states[np.argmax(list_scores)] 
        else:
            best_num_hidden_states = self.n_constant
            
        #best_num_states = list_num_hidden_states[np.argmax(list_scores)] if list_scores else self.n_constant
        
        #print("[SelectorCV] Result model n_compents: {}".format(best_num_hidden_states))
        
        return self.base_model(best_num_hidden_states)
        #raise NotImplementedError
