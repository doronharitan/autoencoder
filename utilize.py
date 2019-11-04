import matplotlib.pyplot as plt

def plot_data(data, compared_data = False):
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    data_len = len(data)
    for row in range(nrows):
        if (row + 1) % 2 == 0 and compared_data:
            pass
        elif (row + 1) % 2 == 0:
            for col in range(ncols):
                ax[row,col].imshow(data[data_len - col - 1])
        else:
            for col in range(ncols):
                ax[row,col].imshow(data[col])
    plt.show()

