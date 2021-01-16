from time import sleep, time

from dateutil.parser import parse
from requests import get

from authorize import get_token

track_analysis_endpoint = 'https://api.spotify.com/v1/audio-features'


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


def _get_track_features_url():
    return f'{track_analysis_endpoint}'


def n_track_features(track_ids):
    if len(track_ids) == 0:
        return []
    token = get_token()
    headers = {'Authorization': f"{token['token_type']} {token['access_token']}"}
    while True:
        response = get(_get_track_features_url(), headers=headers, params={'ids': ','.join(track_ids)})
        if response.status_code == 429:
            print(f'we were rate limited')
        if response.status_code == 200:
            return response.json()['audio_features']
        elif 'Retry-After' in headers:
            print(f"Got status code {response.status_code} with a Retry-After header. Retrying after {headers['Retry-After']}")
            process_retry(headers['Retry-After'])
        else:
            print(response)
            print(response.reason)
            raise Exception("Unexpected non-200 status code")
