import numpy as np
from hmmlearn import hmm
import copy


class HMM_model:

    model   = None
    mixture = False
    
    def train(self, training_data_dict: dict, verbose=False):
        if verbose:
            print("Formatting training data...")
        minor_sequences, minor_sequence_lengths, major_sequences, major_sequence_lengths = self.format_training_data(training_data_dict)
        if verbose:
            print("Done.")
        self.model = self.train_model(minor_sequences, minor_sequence_lengths, major_sequences, major_sequence_lengths, hidden_states=3, iterations=100, verbose=verbose)
        return
    
    def predict(self, test_sample: dict, mode=False):
        seq = self.format_sequence(test_sample)
        if mode is False:
            estimate = np.argmax([mdl.score(seq) for mdl in self.model])
        elif mode == 0:
            estimate = np.argmax([mdl.score(seq) for mdl in self.model[:12]])
        else:
            estimate = np.argmax([mdl.score(seq) for mdl in self.model[12:]]) + 12
        return estimate

    
    def train_model(self, minor_sequences, minor_sequence_lengths, major_sequences, major_sequence_lengths, hidden_states, iterations, verbose=False):

        # Train two base models for major and minor
        if self.mixture:
            model_minor = hmm.GMMHMM(n_components=hidden_states, covariance_type="full", n_iter=iterations, n_mix=10, verbose=verbose)
            model_major = hmm.GMMHMM(n_components=hidden_states, covariance_type="full", n_iter=iterations, n_mix=10, verbose=verbose)
        else:
            model_minor = hmm.GaussianHMM(n_components=hidden_states, covariance_type="full", n_iter=iterations, verbose=verbose)
            model_major = hmm.GaussianHMM(n_components=hidden_states, covariance_type="full", n_iter=iterations, verbose=verbose)
        
        if verbose:
            print("Training minor model...")
        model_minor.fit(minor_sequences, minor_sequence_lengths)
        if verbose:
            print("Trained minor model. Converged: %s" % str(model_minor.monitor_.converged))
            print("Training major model...")
        model_major.fit(major_sequences, major_sequence_lengths)
        if verbose:
            print("Trained major model. Converged: %s" % str(model_major.monitor_.converged))
            print("Done.")

        # Transpose the base models to all other keys
        if verbose:
            print("Copying models...")
        models = []
        for i in range(0, 12):
            key_model = copy.deepcopy(model_minor)
            if self.mixture:
                key_model.means_ = np.roll(key_model.means_, i, axis=2)
                key_model.covars_ = np.roll(np.roll(key_model.covars_, i, axis=2), i, axis=3)
            else:
                key_model.means_ = np.roll(key_model.means_, i, axis=1)
                key_model.covars_ = np.roll(np.roll(key_model.covars_, i, axis=1), i, axis=2)
            models.append(key_model)
        for i in range(0, 12):
            key_model = copy.deepcopy(model_major)
            if self.mixture:
                key_model.means_ = np.roll(key_model.means_, i, axis=2)
                key_model.covars_ = np.roll(np.roll(key_model.covars_, i, axis=2), i, axis=3)
            else:
                key_model.means_ = np.roll(key_model.means_, i, axis=1)
                key_model.covars_ = np.roll(np.roll(key_model.covars_, i, axis=1), i, axis=2)
            models.append(key_model)
        if verbose:
            print("Done")
        return models

    def format_training_data(self, training_data_dict: dict):
        minor_sequences        = np.zeros((0,12))
        minor_sequence_lengths = []
        major_sequences        = np.zeros((0,12))
        major_sequence_lengths = []
        for track_id in training_data_dict:
            analysis = training_data_dict[track_id]
            
            # Format sequence
            seq = self.format_sequence(analysis)
            seq = np.roll(seq, -analysis["key"]) # All sequences are transposed to the "base" key (C)
            
            # Add sequence to set of all sequences within mode
            if analysis["mode"] == 1:
                major_sequences = np.concatenate((major_sequences, seq))
                major_sequence_lengths.append(seq.shape[0])
            else:
                minor_sequences = np.concatenate((minor_sequences, seq))
                minor_sequence_lengths.append(seq.shape[0])

        minor_sequence_lengths = np.array(minor_sequence_lengths)
        major_sequence_lengths = np.array(major_sequence_lengths)
        return minor_sequences, minor_sequence_lengths, major_sequences, major_sequence_lengths
    
    def format_sequence(self, audio_analysis):
        return audio_analysis["pitches"]