class AwareDefaultDict(dict):
    def __init__(self, create_using_key):
        self._obj_dict = dict()
        self._create_using_key = create_using_key

    def __getitem__(self, key):
        if key not in self._obj_dict:
            new_instance = self._create_using_key(key)
            self[key] = new_instance

        return self._obj_dict[key]

    def __setitem__(self, key, value):
        self._obj_dict[key] = value