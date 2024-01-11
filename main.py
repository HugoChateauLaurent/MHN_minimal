import torch
import torchvision
import matplotlib.pyplot as plt

from mhn import ModernHopfieldNetwork

BATCH_SIZE = 32
MEMORY_SIZE = 1000
PATTERN_SIZE = 28**2
BETA = .05
BETA2 = 1e5

def main():

    torch.manual_seed(0)
    device = "cpu" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Try storing random data
    keys = torch.rand((MEMORY_SIZE, PATTERN_SIZE)).float().to(device)
    queries = torch.rand((BATCH_SIZE, PATTERN_SIZE)).float().to(device)

    memory = ModernHopfieldNetwork(keys)

    recall = memory(queries, BETA)
    print(recall.shape)



    # Try storing MNIST data
    mnist = torchvision.datasets.MNIST('./data', download=True)
    imgs = mnist.data[:MEMORY_SIZE].reshape(MEMORY_SIZE, PATTERN_SIZE).float().to(device)

    memory = ModernHopfieldNetwork(imgs)

    for i in range(5):

        query = imgs[i:i+1].clone()

        fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(12,4))
        fig.suptitle("Example "+str(i+1)+"/5")

        ax[0].matshow(query.reshape(28,28), cmap="gray")
        ax[0].title.set_text("Original")

        query += torch.normal(mean=0, std=200, size=query.shape) # add noise to query to make it harder for the network to recall the memory
        ax[1].matshow(query.reshape(28,28), cmap="gray")
        ax[1].title.set_text("Query")

        recall = memory(query, BETA)
        ax[2].matshow(recall.reshape(28,28), cmap="gray")
        ax[2].title.set_text("Recall with\nBETA="+str(BETA))

        recall = memory(query, BETA2)
        ax[3].matshow(recall.reshape(28,28), cmap="gray")
        ax[3].title.set_text("Recall with\nBETA="+str(BETA2))

        recall = memory(query, BETA, sample=True, device=device)
        ax[4].matshow(recall.reshape(28,28), cmap="gray")
        ax[4].title.set_text("Recall with\nBETA="+str(BETA)+"\n(sampling)")

        recall = memory(query, BETA2, sample=True, device=device)
        ax[5].matshow(recall.reshape(28,28), cmap="gray")
        ax[5].title.set_text("Recall with\nBETA="+str(BETA2)+"\n(sampling)")

        for a in ax:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)

        plt.show()

if __name__ == "__main__":
    main()