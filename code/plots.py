import cv2
import numpy as np
import matplotlib.pyplot as plt

N_MAX_IMAGES = 50

def plot_images(images, titles = None, title_fig_window = None, fig_size = (15, 15), font_size = 12, show_ticks = False, n_max_cols = 4):

    n_images = len(images)

    if n_images > N_MAX_IMAGES:
        print('ERROR:show_images(): You are trying to show ' + str(n_images) + ' but the maximum is ' + str(N_MAX_IMAGES) + '!')
        return

    n_cols = int(min(n_images, n_max_cols))
    n_rows = int(np.ceil(n_images / n_cols))

    plt.figure(title_fig_window, figsize = fig_size)

    for i in range(n_images):
        
        plt.subplot(n_rows, n_cols, i + 1)

        image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        
        plt.imshow(image)
        
        if titles is not None:
            plt.title(titles[i], fontsize = font_size)
        if not show_ticks:
            plt.xticks([])
            plt.yticks([])

    plt.show()