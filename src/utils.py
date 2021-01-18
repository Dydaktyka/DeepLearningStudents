import itertools
import matplotlib.pyplot as plt
import time

def display_img_grid(rows, cols, imgs, filename=None):
    fig, ax = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))
    for i, j in itertools.product(range(rows), range(cols)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
        ax[i, j].imshow(imgs[i * cols + j], cmap='Greys')
    if filename:
        plt.savefig(filename);
        plt.close();


def estimate(start, n_epochs, epochs):
    end = time.time()
    ellapsed = end - start
    time_per_epoch = ellapsed / epochs
    remaining = time_per_epoch * (n_epochs - epochs)
    return ellapsed, remaining
