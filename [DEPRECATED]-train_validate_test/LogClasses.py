# non-throwing messages
class GenericMessage():
    def __init__(self, msg: str, prefix: str):
        self.msg = msg
        self.prefix = prefix

    def __str__(self):
        return f"{self.prefix} {self.msg}"
    
class InfoMessage(GenericMessage):
    def __init__(self, msg: str):
        super().__init__(msg, "[INFO]")

class WarningMessage(GenericMessage):
    def __init__(self, msg: str):
        super().__init__(msg, "[WARNING]")

# throwing messages
class GenericException(Exception):
    def __init__(self, msg: str, prefix: str):
        self.msg = msg
        self.prefix = prefix

    def __str__(self):
        return f"{self.prefix} {self.msg}"

class NoCheckpointException(GenericException):
    def __init__(self, msg: str = "No checkpoint provided for training"):
        super().__init__(msg, "[ERROR]")

class NoTrainingDataException(GenericException):
    def __init__(self, msg: str = "No training data provided"):
        super().__init__(msg, "[ERROR]")

class NoTrainingArgumentException(GenericException):
    def __init__(self, msg: str = "No training arguments provided"):
        super().__init__(msg, "[ERROR]")
