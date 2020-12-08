from os.path import exists, join
from os import getcwd, makedirs
from pickle import load, dump
from requests import get
from authorize import get_token
from time import sleep, time
from dateutil.parser import parse

track_analyses_folder = 'track_analyses'
track_analysis_endpoint = 'https://api.spotify.com/v1/audio-analysis'


def isfloat(flt):
    try:
        float(flt)
        return True
    except ValueError:
        return False


def process_retry(retry_after):
    # It is either a float  or an HTTP date
    if not isfloat(retry_after):
        retry_after = parse(retry_after).timestamp() - time()
    sleep(retry_after)


def _get_track_analysis_url(track_id):
    return f'{track_analysis_endpoint}/{track_id}'


def _get_track_path(track_id):
    return join(getcwd(), track_analyses_folder, track_id)


def _track_analysis_exists(track_id):
    return exists(_get_track_path(track_id))


def _store_track_analysis(track_analysis, track_id):
    print(f'storing track analysis for {track_id}')
    print(track_analysis)
    with open(_get_track_path(track_id), 'wb') as f:
        dump(track_analysis, f)


def _load_track_analysis(track_id):
    print(f'loading track analysis for {track_id}')
    with open(_get_track_path(track_id), 'rb') as f:
        return load(f)


def fetch_track_analysis(track_id):
    token = get_token()
    headers = {'Authorization': f"{token['token_type']} {token['access_token']}"}
    response = get(_get_track_analysis_url(track_id), headers=headers)
    result = response.json()
    result['track']['id'] = track_id
    return result


def n_track_analyses_generator(track_ids):
    token = get_token()
    headers = {'Authorization': f"{token['token_type']} {token['access_token']}"}
    idx = 0
    start_time = time()
    while True:
        if idx == len(track_ids):
            break
        print(f'looking for track {idx} out of {len(track_ids)}')
        track_id = track_ids[idx]
        response = get(_get_track_analysis_url(track_id), headers=headers)
        if response.status_code == 429:
            print(f'we were rate limited after {idx + 1} requests')
        if response.status_code == 200:
            result = response.json()
            result['track']['id'] = track_id
            yield result
            idx += 1
        elif 'Retry-After' in headers:
            print(f"Got status code {response.status_code} with a Retry-After header. Retrying after {headers['Retry-After']}")
            process_retry(headers['Retry-After'])
        else:
            print(response)
            raise Exception("Unexpected non-200 status code")
    total_time = time() - start_time
    print(f'fetched {len(track_ids)} track analysis objects in {total_time:.3f} seconds')


def get_track_analysis(track_id):
    if not _track_analysis_exists(track_id):
        track_analysis = fetch_track_analysis(track_id)
        _store_track_analysis(track_analysis, track_id)
        return track_analysis
    return _load_track_analysis(track_id)


if not exists(join(getcwd(), track_analyses_folder)):
    makedirs(join(getcwd(), track_analyses_folder))
