import numpy as np

def extract_audio_features(audio_features):
    return {
        "id": audio_features['id'],
        "key": audio_features['key'],
        "mode": audio_features['mode']
    }

def extract_track_analysis(track_analysis):
    track_object = {
        "id": track_analysis['track']['id'],
        "duration": track_analysis['track']['duration'],
        "key": track_analysis['track']['key'],
        "key_confidence": track_analysis['track']['key_confidence'],
        "mode": track_analysis['track']['mode'],
        "mode_confidence": track_analysis['track']['mode_confidence']
    }
    track_object['pitches'], track_object['start'], track_object['duration'], track_object['confidence'] =\
        format_first_n_seconds_segments(track_analysis['segments'], 20)
    return track_object

def format_first_n_seconds_segments(segments, secs):
    ''' Transforms the chroma vectors, start, duration and condifidence values for each
    segment to numpy arrays up to a certain initial amount of seconds.
    '''
    num_segments = len(segments)
    start_vec       = np.zeros(num_segments)
    duration_vec    = np.zeros(num_segments)
    confidence_vec  = np.zeros(num_segments)
    chroma_vecs     = np.zeros( (num_segments, 12) )
    idx = 0
    for segment in segments:
        start_vec[idx] = segment['start']
        duration_vec[idx] = segment['duration']
        confidence_vec[idx] = segment['confidence']
        chroma_vecs[idx,:] = segment["pitches"]
        idx += 1
        if segment['start'] + segment['duration'] >= secs:
            break
    return chroma_vecs[:idx,:], start_vec[:idx], duration_vec[:idx], confidence_vec[:idx]


def extract_segment(segment):
    return {
        'start': segment['start'],
        'duration': segment['duration'],
        'confidence': segment['confidence'],
        'pitches': segment['pitches']
    }
