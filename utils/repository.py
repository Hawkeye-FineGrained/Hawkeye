class Repository(dict):
    """
    A dict format repository to register module.
    Repository can also manage config node.
    """

    def __init__(self, *args, **kwargs):
        super(Repository, self).__init__(*args, **kwargs)

    def register(self, module):
        assert module.__name__ not in self
        self[module.__name__] = module
        return module
