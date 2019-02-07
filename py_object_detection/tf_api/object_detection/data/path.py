from os.path import dirname, realpath, exists


def get(file_name=None):
    if not file_name:
        return dirname(realpath(__file__))
    path = get() + "/" + file_name
    if exists(path):
        return path
    return None
