class Center():
    r"""Centers node positions :obj:`pos` around the origin."""
    def __call__(self, data):
        data = data - data.mean(dim=-2, keepdim=True)
        return data

class NormalizeScale():
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """
    def __init__(self):
        self.center = Center()

    def __call__(self, data):
        data = self.center(data)

        scale = (1 / data.abs().max()) * 0.999999
        data = data * scale

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)