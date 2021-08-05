class Config():
    def __init__(self):
        pass

    num_classes = 1
    labels_to_class = {0: 'hem',1:'all'}
    class_to_labels = {'hem':0,'all' : 1}
    resize = 300
    num_epochs = 3
    batch_size = 100