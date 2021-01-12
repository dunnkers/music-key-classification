import requests
import dill as pickle
import os
import os.path as path
import time


def get_client_id():
    return "1459c33d0d8f4e90b75fabbc23d36187"


def get_client_secret():
    return "fd26fca84ba54d1db78a37f157d6b861"


def get_pickle_file_name():
    return path.join(os.getcwd(), 'bearer-token')


def store_token(token):
    # Time we received it (in secs) + time it lasts (in secs) is time it expires (in secs)
    token['expire_time'] = time.time() + token['expires_in']
    with open(get_pickle_file_name(), 'wb') as f:
        pickle.dump(token, f)


def load_token():
    with open(get_pickle_file_name(), 'rb') as f:
        return pickle.load(f)


def fetch_new_token():
    payload = {'grant_type': 'client_credentials'}
    r = requests.post('https://accounts.spotify.com/api/token', data=payload,
                      auth=(get_client_id(), get_client_secret()))
    return r.json()


def get_token():
    if (path.lexists(get_pickle_file_name())):
        token = load_token()
        if time.time() < (token['expire_time'] - 0.1):
            return token
    token = fetch_new_token()
    store_token(token)
    return token
