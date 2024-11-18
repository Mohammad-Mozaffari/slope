class EmptyBackend():
    def size(self):
        return 1
    def rank(self):
        return 0

    def broadcast(self, tensor, src, async_op=True):
        return None