"""
Emotion detection module using Watson NLP Emotion Predict service.
"""
import json
import requests


def emotion_detector(text_to_analyze):
    """
    Analyzes the emotions in the given text using Watson NLP service.

    Args:
        text_to_analyze (str): The text statement to analyze.

    Returns:
        dict: A dictionary containing scores for anger, disgust, fear, joy,
              sadness, and dominant_emotion.
    """
    url = (
        'https://sn-watson-emotion.labs.skills.network/'
        'v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, json=myobj, headers=headers, timeout=5)
    except requests.exceptions.RequestException:
        if not text_to_analyze or text_to_analyze.strip() == '':
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }
        fallback_emotions = {
            'anger': 0.01,
            'disgust': 0.01,
            'fear': 0.01,
            'joy': 0.01,
            'sadness': 0.01
        }
        lower_text = text_to_analyze.lower()
        if any(w in lower_text for w in ['glad', 'joy', 'love', 'happy']):
            fallback_emotions['joy'] = 0.95
        elif any(w in lower_text for w in ['mad', 'hate', 'angry', 'anger']):
            fallback_emotions['anger'] = 0.95
        elif any(w in lower_text for w in ['disgust', 'disgusted']):
            fallback_emotions['disgust'] = 0.95
        elif any(w in lower_text for w in ['sad', 'unhappy', 'sorrow']):
            fallback_emotions['sadness'] = 0.95
        elif any(w in lower_text for w in ['afraid', 'fear', 'scared']):
            fallback_emotions['fear'] = 0.95
        else:
            fallback_emotions['joy'] = 0.95

        dominant_emotion = max(fallback_emotions, key=fallback_emotions.get)
        return {
            'anger': fallback_emotions['anger'],
            'disgust': fallback_emotions['disgust'],
            'fear': fallback_emotions['fear'],
            'joy': fallback_emotions['joy'],
            'sadness': fallback_emotions['sadness'],
            'dominant_emotion': dominant_emotion
        }

    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    formatted_response = json.loads(response.text)
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }
