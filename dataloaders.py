def loadfile(path: str):
    with open(path, "r") as f:
        while f.readable():
            yield [eval(v) for v in f.readline().split(" ")]