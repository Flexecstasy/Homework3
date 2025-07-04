import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(history, save_path=None):
    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_heatmap(results_matrix, x_labels, y_labels, save_path=None):
    plt.figure()
    plt.imshow(np.array(results_matrix), interpolation='nearest')
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.close()