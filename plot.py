import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from matplotlib.animation import FuncAnimation


def plot(X, y, y_preds):
    fig = plt.figure(figsize=(12, 6))
    plt.title("Predictions")
    plt.grid()
    plt.xlim((-4, 4))
    plt.ylim((-1.5, 1.5))

    plt.scatter(
        X.value.flatten(),
        y.value.flatten(),
        s=0.5,
        color="C0",
        label="true y",
    )

    line_plotted = plt.scatter(
        X.value.flatten(),
        y_preds[0],
        s=0.5,
        color="C1",
        label="pred y",
    )
    plt.legend()

    def animation_function(frame):
        data = np.stack((X.value.flatten(), y_preds[frame])).T
        line_plotted.set_offsets(data)
        plt.title(f"Epoch {frame * 100}")

    anim_created = FuncAnimation(
        fig, animation_function, frames=len(y_preds), interval=100
    )
    video = anim_created.to_html5_video()
    html = display.HTML(video)
    display.display(html)

    plt.close()


def plot_one(X, y, y_preds):
    fig = plt.figure(figsize=(12, 6))
    plt.xlim((-4, 4))
    plt.ylim((-1.5, 1.5))

    plt.scatter(
        X.value.flatten(),
        y.value.flatten(),
        s=0.5,
        color="C0",
    )

    plt.scatter(
        X.value.flatten(),
        y_preds[-1],
        s=0.5,
        color="C1"
    )