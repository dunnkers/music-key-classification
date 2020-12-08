from authorize import get_token
from requests import get

# track_analyses_folder = 'track_analyses'
track_analysis_endpoint = 'https://api.spotify.com/v1/tracks'
tracks_fetch_limit = 50


def _get_tracks_url(_id):
    return f'{track_analysis_endpoint}/{_id}'


def fetch_track(track_id):
    token = get_token()
    headers = {'Authorization': f"{token['token_type']} {token['access_token']}"}
    response = get(_get_tracks_url(track_id), headers=headers)
    return response.json()


def fetch_tracks(track_ids):
    if len(track_ids) > tracks_fetch_limit:
        raise Exception(f"Cannot fetch more than {tracks_fetch_limit} tracks at a time")
    token = get_token()
    headers = {'Authorization': f"{token['token_type']} {token['access_token']}"}
    data = {"ids": ','.join(track_ids)}
    response = get(_get_tracks_url(''), params=data, headers=headers)
    return response.json()
