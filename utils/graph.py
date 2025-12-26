import matplotlib.pyplot as plt

def plot_history(history: dict) -> plt:
    loss = history["loss"]
    val_loss = history["val_loss"]
    f1 = history["f1"]
    val_f1 = history["val_f1"]
    epoch = len(loss)
    x_max = range(epoch)

    plt.figure(1, figsize=(7,5))
    plt.plot(x_max, loss)
    plt.plot(x_max, val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss & val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.style.use(['classic'])

    plt.figure(2, figsize=(7,5))
    plt.plot(x_max, f1)
    plt.plot(x_max, val_f1)
    plt.xlabel('epoch')
    plt.ylabel('f1')
    plt.title('f1 & val_f1')
    plt.grid(True)
    plt.legend(['train','val'], loc=4)
    plt.style.use(['classic'])

    return plt