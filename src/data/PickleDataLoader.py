import pickle
import random

import time

from data.DataWindow import DataWindow

DATA_KEY = 'metric2_list'


class PickleDataLoader:
    """
        :arg
        fileName : the name of the file which contains the data, in pickle format and the key of data is 'metric2_list'
        originN : how many data points are in a DataWindow for prediction
        predN : how many data points are in a DataWindow for verification
        stride : the interval between two consecutive DataWindows
    """

    def __init__(self, file_name, origin_num, pred_num, stride) -> None:
        super().__init__()
        self.file_name = file_name
        self.originN = origin_num
        self.predN = pred_num
        self.stride = stride

    def load(self, train_ratio):
        pickle_obj = pickle.load(open(self.file_name, 'rb'))
        data_list = pickle_obj[DATA_KEY]
        data_list = self.preprocess(data_list)
        random.Random(time.time()).shuffle(data_list)

        index = 0
        data_len = len(data_list)
        print('The data size is %d' % data_len)
        train_list = []
        test_list = []
        while index + self.predN + self.originN < int(data_len*train_ratio):
            window = DataWindow(data_list[index:index + self.originN],
                                data_list[index + self.originN: index + self.originN + self.predN])
            train_list.append(window)
            index += self.stride
        while index + self.predN + self.originN < data_len:
            window = DataWindow(data_list[index:index + self.originN],
                                data_list[index + self.originN: index + self.originN + self.predN])
            test_list.append(window)
            index += self.stride
        return train_list, test_list

    def preprocess(self, data_list):
        ret_list = []
        data_len = len(data_list)
        for i in range(data_len):
            ret_list.append(data_list[i] / data_list[0])
        return ret_list


if __name__ == '__main__':
    loader = PickleDataLoader('/home/cdx4838/PycharmProjects/exp2/exp2/data/cnn1.log.cache', 10, 2, 1)
    alist = loader.load()
    print(len(alist))
    print(alist)
