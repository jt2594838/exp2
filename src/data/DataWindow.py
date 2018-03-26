
class DataWindow:
    """
        :arg
        originD : the data used for prediction
        nextD : the data to verify prediction result
    """
    def __init__(self, originD, nextD) -> None:
        super().__init__()
        self.originD = originD
        self.nextD = nextD
