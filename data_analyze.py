import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


def lesion_distribution(metadata):
    lesion_type_amount = {k : len(metadata[metadata["dx"] == k]) for k in metadata["dx"].unique()}
    print(lesion_type_amount)
    plt.bar(x=list(range(7)), height=lesion_type_amount.values(), tick_label=list(lesion_type_amount.keys()))
    plt.title("Rozkład chorób")
    plt.savefig("diagrams/lesion_distributions.png")


def lesion_example(metadata):
    n_samples = 5

    fig, m_axs = plt.subplots(7, n_samples, figsize = (n_samples, 7))
    #fig.set_title("Przykłady znamion")
    for n_axs, lesion_type in zip(m_axs, metadata["ilness"].unique()):
        type_rows = metadata[metadata["ilness"] == lesion_type]
        n_axs[2].set_title(lesion_type)
        for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
            img = imread("data/ham10000_images/"+ c_row["image_id"] + ".jpg")
            c_ax.imshow(img)
            c_ax.axis('off')
    plt.show()
    plt.savefig("diagrams/lesion_example.jpg")



if __name__ == "__main__":
    metadata = pd.read_csv("data/HAM10000_metadata.csv")
    print(metadata.columns, metadata.head())

    metadata["ilness"] = metadata["dx"].map(lesion_type_dict.get)

    lesion_distribution(metadata)
    lesion_example(metadata)




