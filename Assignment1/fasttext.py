import os
from fastText import train_supervised


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

if __name__ == "__main__":
    train_data = os.path.join(os.getenv("DATADIR", ''), 'posneg.train.txt')
    valid_data = os.path.join(os.getenv("DATADIR", ''), 'test.txt')

    # train_supervised uses the same arguments and defaults as the fastText cli
    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1
    )
    print_results(*model.test(valid_data))

    model.save_model("posneg.bin")

    texts = ['He likes Indian food.', 'He does not eat junk food.']
    labels = model.predict(texts)
    print (labels)
