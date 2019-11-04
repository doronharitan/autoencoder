import matplotlib.pyplot as plt

def plot_data(data, compared_data=[]):
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10), sharey=True)
    data_len = len(data)
    for row in range(nrows):
        if (row + 1) % 2 == 0 and len(compared_data) > 0:
            for col in range(ncols):
                ax[row,col].imshow(compared_data[col])
                ax[row, col].set_title('After ' + str(col))
        elif (row + 1) % 2 == 0:
            for col in range(ncols):
                ax[row,col].imshow(data[data_len - col - 1])
                ax[row, col].set_title('Before ' + str(data_len - col - 1))
        else:
            for col in range(ncols):
                ax[row,col].imshow(data[col])
                ax[row, col].set_title('Before ' + str(col))
    if len(compared_data) == 0:
        plt.savefig('untouched_data', bbox_inches="tight")
    else:
        plt.savefig('before_vs_after', bbox_inches="tight")
