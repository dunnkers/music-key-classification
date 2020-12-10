import logging
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import pandas as pd
import time
import sys

# Spotify credentials are to be put in creds.py. Checking for that.
try:
    import creds
except:
    logging.error("There does not seem to be a creds.py file containing a Spotify API key/secret pair.")
    sys.exit()
if creds.key == '' or creds.secret == '':
    logging.error("Please enter a Spotify API key/secret pair in creds.py.")
    sys.exit()


# CSV file with Spotify song IDs that I pulled from internet somewhere
data_file    = "data.csv"

# File to put the result of this program
out_file     = "out.csv"

# Number of songs to sample - note that only songs that are in major key are considered so
# the final number of samples that are used will be lower (probably about 30% lower)
num_samples  = 200

# The major scales.
key_nums = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
scales = {
    #      C  C#  D  D#  E  F  F#  G  G#  A  A#  B
    'C':  [1, 0,  1, 0,  1, 1, 0,  1, 0,  1, 0,  1],
    'C#': [1, 1,  0, 1,  0, 1, 1,  0, 1,  0, 1,  0],
    'D':  [0, 1,  1, 0,  1, 0, 1,  1, 0,  1, 0,  1],
    'D#': [1, 0,  1, 1,  0, 1, 0,  1, 1,  0, 1,  0],
    'E':  [0, 1,  0, 1,  1, 0, 1,  0, 1,  1, 0,  1],
    'F':  [1, 0,  1, 0,  1, 1, 0,  1, 0,  1, 1,  0],
    'F#': [0, 1,  0, 1,  0, 1, 1,  0, 1,  0, 1,  1],
    'G':  [1, 0,  1, 0,  1, 0, 1,  1, 0,  1, 0,  1],
    'G#': [1, 1,  0, 1,  0, 1, 0,  1, 1,  0, 1,  0],
    'A':  [0, 1,  1, 0,  1, 0, 1,  0, 1,  1, 0,  1],
    'A#': [1, 0,  1, 1,  0, 1, 0,  1, 0,  1, 1,  0],
    'B':  [0, 1,  0, 1,  1, 0, 1,  0, 1,  0, 1,  1]
}

# Retrieves a token - Throws an Exception when unsuccesful
def getToken():
    res = requests.post('https://accounts.spotify.com/api/token', auth=HTTPBasicAuth(creds.key, creds.secret), data={"grant_type": "client_credentials"})
    if res.status_code == 200:
        newToken   = res.json()
        token      = newToken["access_token"]
        expires_in = int(newToken["expires_in"]) - 5 # take some extra time just to be sure
        return token, expires_in
    else:
        raise Exception('Token retrieval failed.')
    return None, None

# Retrieves the audio-analysis object for a given song ID.
# Contains some very basic exception handling, simply returns None on failure.
def getAudioAnalysis(song_id, token):
    request_uri = f'https://api.spotify.com/v1/audio-analysis/{song_id}'
    res = requests.get(request_uri, headers={"Authorization": f"Bearer {token}"})
    if res.status_code == 419:
        wait_sec = int(res.headers["Retry-After"])
        print(f"Was asked to wait {wait_sec} seconds.")
        time.sleep(wait_sec)
        return getAudioAnalysis(song_id, token)
    elif res.status_code == 200:
        return res.json()
    else:
        logging.error(f"Request to GET {request_uri} failed.")
    return None

# Naive key estimation routine. 
# Data is first reformatted to a Numpy array, then processed.
# This method compares the weighted (in this case, by duration) average chroma key
# for every segment with each scale's notes.
def estimate_key(audio_analysis):

    # Reformat data
    num_segments = len(analysis["segments"])
    input_data = np.zeros((num_segments, 13))
    i = 0
    for segment in analysis["segments"]:
        input_data[i,0]  = segment["duration"]
        input_data[i,1:] = segment["pitches"]
        i += 1
    
    # Compare weighted average chroma key with every scale
    avg_vec = np.average(input_data[:,1:], axis=0, weights=input_data[:,0])
    scores = np.zeros(12)
    for i in range(0, 12):
        scores[i] = np.correlate(avg_vec, scales[key_nums[i]])
    return np.argmax(scores)



### MAIN SCRIPT ###

# Get an initial token, open the CSV file for track IDs, initialize some values
token, token_refresh_time = getToken()
latest_token_update = time.time()
test_data = pd.read_csv(data_file)
conf_mat = np.zeros((12,12))


n_perc = int(num_samples/100)
i = 0
for track_sample in test_data.sample(n = num_samples).to_dict(orient = 'records'):

    # Possibly update the token
    if time.time() - latest_token_update > token_refresh_time:
        token, token_refresh_time = getToken()
        print("Token was refreshed.")

    # Add classification to the confusion matrix
    track = track_sample["id"]
    analysis = getAudioAnalysis(track, token)
    if analysis is not None and analysis["track"]["mode"] == 1:
        correct_answer = analysis["track"]["key"]
        estimate = estimate_key(analysis)
        conf_mat[correct_answer, estimate] += 1
    
    i += 1
    if i % n_perc == 0:
        print(f"{int(i/n_perc)}%")


# Outputting
print(conf_mat)
np.savetxt(out_file, conf_mat, delimiter=",", fmt="%d")
