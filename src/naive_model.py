import numpy as np
from hmmlearn import hmm
import copy

class Naive_model:

    model = -np.ones((24,12))
    model[12,:] = [1, 0,  1, 0,  1, 1, 0,  1, 0,  1, 0,  1]
    model[13,:] = [1, 1,  0, 1,  0, 1, 1,  0, 1,  0, 1,  0]
    model[14,:] = [0, 1,  1, 0,  1, 0, 1,  1, 0,  1, 0,  1]
    model[15,:] = [1, 0,  1, 1,  0, 1, 0,  1, 1,  0, 1,  0]
    model[16,:] = [0, 1,  0, 1,  1, 0, 1,  0, 1,  1, 0,  1]
    model[17,:] = [1, 0,  1, 0,  1, 1, 0,  1, 0,  1, 1,  0]
    model[18,:] = [0, 1,  0, 1,  0, 1, 1,  0, 1,  0, 1,  1]
    model[19,:] = [1, 0,  1, 0,  1, 0, 1,  1, 0,  1, 0,  1]
    model[20,:] = [1, 1,  0, 1,  0, 1, 0,  1, 1,  0, 1,  0]
    model[21,:] = [0, 1,  1, 0,  1, 0, 1,  0, 1,  1, 0,  1]
    model[22,:] = [1, 0,  1, 1,  0, 1, 0,  1, 0,  1, 1,  0]
    model[23,:] = [0, 1,  0, 1,  1, 0, 1,  0, 1,  0, 1,  1]

    def train(self, training_data_dict: dict):
        teacher_vecs = {}
        for i in range(0,24):
            teacher_vecs[i] = []

        print("Applying training samples...")
        for track_id in training_data_dict:
            track_data = training_data_dict[track_id]
            vec = self.format_sequence(track_data)
            teacher_vecs[track_data["mode"]*12 + track_data["key"]].append(vec)
        print("Done.")
        
        print("Composing model...")
        self.model = []
        for i in range(0,24):
            if len(teacher_vecs[i]) == 0:
                self.model.append(-np.ones(12))
            else:
                self.model.append( np.average(np.array(teacher_vecs[i]), axis=0) )
        print("Done.")
        self.model = np.array(self.model)
        
    
    def predict(self, test_sample: dict):
        avg_vec = self.format_sequence(test_sample)
        scores = np.zeros(24)
        for i in range(0, 24):
            scores[i] = np.correlate(avg_vec, self.model[i])
        return np.argmax(scores)
    
    
    def format_sequence(self, audio_analysis):
        # Reformat data
        num_segments = len(audio_analysis["segments"])
        input_data = np.zeros((num_segments, 13))
        i = 0
        for segment in audio_analysis["segments"]:
            input_data[i,0]  = segment["duration"]
            input_data[i,1:] = segment["pitches"]
            i += 1
        
        # Return weighted average chroma key vector
        avg_vec = np.average(input_data[:,1:], axis=0, weights=input_data[:,0])
        return avg_vec