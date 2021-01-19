def extract_audio_features(audio_features):
    return {
        "id": audio_features['id'],
        "key": audio_features['key'],
        "mode": audio_features['mode']
    }


def extract_track_analysis(track_analysis):
    return {
        "id": track_analysis['track']['id'],
        "segments": [extract_segment(segment) for segment in track_analysis['segments']],
        "duration": track_analysis['track']['duration'],
        "key": track_analysis['track']['key'],
        "key_confidence": track_analysis['track']['key_confidence'],
        "mode": track_analysis['track']['mode'],
        "mode_confidence": track_analysis['track']['mode_confidence']
    }


def extract_segment(segment):
    return {
        'start': segment['start'],
        'duration': segment['duration'],
        'confidence': segment['confidence'],
        'pitches': segment['pitches']
    }
