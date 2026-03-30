import os
import csv
import numpy as np


def save_history_csv(history, output_dir):
    path = os.path.join(output_dir, 'training_log.csv')
    keys = history[0].keys()
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"  Saved: {path}")


def save_plots(history, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots")
        return

    epochs = [h['epoch'] for h in history]

    # Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['train_loss'] for h in history], 'b-o', markersize=3, label='Train Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['train_acc'] for h in history], 'b-o', markersize=3, label='Train Acc')
    ax.plot(epochs, [h['test_acc'] for h in history], 'r-s', markersize=3, label='Test Acc')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Train vs Test Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=150)
    plt.close(fig)

    # F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['test_f1w'] for h in history], 'g-^', markersize=3, label='F1 (weighted)')
    ax.plot(epochs, [h['test_f1m'] for h in history], 'm-v', markersize=3, label='F1 (macro)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score'); ax.set_title('Test F1 Scores')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'f1_curve.png'), dpi=150)
    plt.close(fig)

    print(f"  Saved: loss_curve.png, accuracy_curve.png, f1_curve.png")


def save_confusion_matrix(cm, classes, output_dir, tag):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True', xlabel='Predicted',
           title=f'Confusion Matrix ({tag})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, f'{cm_arr[i, j]}',
                    ha='center', va='center',
                    color='white' if cm_arr[i, j] > thresh else 'black',
                    fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'confusion_matrix_{tag}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: confusion_matrix_{tag}.png")
