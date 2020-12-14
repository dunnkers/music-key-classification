def extract_track_analysis(track_analysis):
    return {
        "id": track_analysis['track']['id'],
        "segments": [extract_segment(segment) for segment in track_analysis['segments']],
        "duration": track_analysis['track']['duration'],
        "key": track_analysis['track']['key'],
        "mode": track_analysis['track']['mode']
    }


def extract_segment(segment):
    return {
        'start': segment['start'],
        'duration': segment['duration'],
        'confidence': segment['confidence'],
        'pitches': segment['pitches']
    }
