import torch


class DataWindow:
    """
        :arg
        originD : the data used for prediction
        nextD : the data to verify prediction result
    """
    def __init__(self, originD, nextD) -> None:
        super().__init__()
        if isinstance(originD, torch.FloatTensor):
            self.originD = originD
        else:
            self.originD = torch.FloatTensor(originD).view(1, -1)
        if isinstance(nextD, torch.FloatTensor):
            self.nextD = nextD
        else:
            self.nextD = torch.FloatTensor(nextD).view(1, -1)
