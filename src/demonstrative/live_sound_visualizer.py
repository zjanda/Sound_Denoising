import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

def visualize_live_microphone_last_n_seconds(window_seconds=5, sample_rate=16000):
    """
    Continuously visualize the last `window_seconds` of live microphone input.

    Args:
        window_seconds (int): Number of seconds to display in the live plot.
        sample_rate (int): Sampling rate in Hz.
    """
    import queue
    import threading

    buffer_size = window_seconds * sample_rate
    audio_buffer = np.zeros(buffer_size, dtype='float32')
    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=1024
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 4))
    time_axis = np.linspace(-window_seconds, 0, buffer_size)
    line, = ax.plot(time_axis, audio_buffer)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Live Microphone Input (Last {window_seconds} Seconds)")
    ax.set_ylim(-1, 1)
    ax.set_xlim(-window_seconds, 0)
    plt.tight_layout()

    print(f"Visualizing live microphone input (showing last {window_seconds} seconds)... Press Ctrl+C to stop.")
    try:
        with stream:
            while True:
                try:
                    data = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                data = data.flatten()
                audio_buffer = np.roll(audio_buffer, -len(data))
                audio_buffer[-len(data):] = data
                line.set_ydata(audio_buffer)
                fig.canvas.draw()
                fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("Stopped live visualization.")
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    visualize_live_microphone_last_n_seconds(window_seconds=5, sample_rate=16000)
