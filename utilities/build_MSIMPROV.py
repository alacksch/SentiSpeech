def buildmspimprov(msp_path):
    import os
    import pandas as pd

    emotion_map = {
        'F01': 'neutral',
        'F02': 'happy',
        'F03': 'sad',
        'F04': 'angry',
    }

    desired_emotions = {'neutral', 'happy', 'sad', 'angry'}
    data = []

    for root, _, files in os.walk(msp_path):
        for filename in files:
            if not filename.endswith('.wav'):
                continue

            parts = filename.split('-')
            if len(parts) < 5:
                print(f"[WARN] Skipping malformed filename: {filename}")
                continue

            emotion_code = parts[3]
            emotion = emotion_map.get(emotion_code)
            if emotion not in desired_emotions:
                continue

            actor_part = parts[2]
            actor_id = int(''.join(filter(str.isdigit, actor_part)))

            full_path = os.path.join(root, filename)
            data.append([full_path, emotion, actor_id])
    return pd.DataFrame(data, columns=["relative_path", "emotion", "actor_id"])
