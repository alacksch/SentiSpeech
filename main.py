from utility.buildravdess import buildravdess
import matplotlib.pyplot as plt

from utility.features import preprocess_audio


def plot_random_audio_examples(df, num_samples=5, duration=2.5, sample_rate=22050, top_db=30):
    plt.figure(figsize=(10, 6))

    # Randomly sample `num_samples` from the dataset
    random_samples = df.sample(n=num_samples)

    # Loop through and plot `num_samples` audio examples
    for i, (_, row) in enumerate(random_samples.iterrows()):
        sample_path = row['relative_path']
        emotion = row['emotion']

        y = preprocess_audio(sample_path, duration=duration, sample_rate=sample_rate, top_db=top_db)

        # Plot the waveform
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(y)
        plt.title(f"Emotion: {emotion} - Sample {i + 1}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_path = "./audio_speech_actors_01-24"
    df = buildravdess(dataset_path)

    plot_random_audio_examples(df, num_samples=5)