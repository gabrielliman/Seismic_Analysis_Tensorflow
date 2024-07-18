import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def visualize_prediction(num_classes,image, true_mask, predicted_mask):
    true_mask=true_mask.squeeze()
    fig, axes = plt.subplots(1, 3, figsize=(5, 5), sharey=True)

    predicted_mask = np.argmax(predicted_mask[0], axis=-1)
    accuracy = np.mean(true_mask == predicted_mask) * 100
    class_colors = plt.cm.viridis(np.linspace(0, 1, num_classes))[:, :3]
    class_labels = [f'Classe {cls}' for cls in range(len(class_colors))]

    #Ground Truth
    overlay_true_mask = np.zeros(image.squeeze().shape + (3,), dtype=np.float32)
    for cls, color in enumerate(class_colors):
        overlay_true_mask[true_mask == cls] = color
    axes[1].imshow(overlay_true_mask)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    #Prediction
    overlay_predicted_mask = np.zeros(image.squeeze().shape + (3,), dtype=np.float32)
    for cls, color in enumerate(class_colors):
        overlay_predicted_mask[predicted_mask == cls] = color
    axes[2].imshow(overlay_predicted_mask)
    axes[2].set_title(f"Prediction\nAccuracy: {accuracy:.2f}%")
    axes[2].axis('off')


    #Legend
    legend_axe = axes[0]
    legend_axe.axis('off')
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=10)
                       for color, label in zip(class_colors, class_labels)]
    legend_axe.legend(handles=legend_elements, loc='lower left')