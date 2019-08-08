class MultiDimensionDict(dict):
    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            super().__setitem__(item, MultiDimensionDict())
            return super().__getitem__(item)
