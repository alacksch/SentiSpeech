import os
import pandas as pd


def buildravdess(ravdess_path):
    emotion_map = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fear',
        7: 'disgust',
        8: 'surprise'
    }

    data = []

    for actor_folder in sorted(os.listdir(ravdess_path)): # actor level loop
        actor_path = os.path.join(ravdess_path, actor_folder)
        actor_id = int(actor_folder.split('_')[1])  # split into a list of something like ["Actor", "01"], it removes the _ for each actor, and we take index 1

        for filename in sorted(os.listdir(actor_path)): # in-actor level loop
            if filename.endswith('.wav'):
                parts = filename.split('-')
                emotion_code = int(parts[2]) # since emotion is the third number in the filename and it splits the filename into a list of all the information
                emotion = emotion_map[emotion_code]
                full_path = os.path.join(ravdess_path, actor_folder, filename)

                data.append([full_path, emotion, actor_id])

    return pd.DataFrame(data, columns=["relative_path", "emotion", "actor_id"])
