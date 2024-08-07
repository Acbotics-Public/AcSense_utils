import logging

logger = logging.getLogger(__name__)


class Generic_Data:
    def __init__(self, header, raw_data):
        self.header = header
        self.raw = raw_data
        self.data = None
        self._parse()

    @classmethod
    def from_header(cls, header, raw_data):
        # id1 = header["Header"].id1
        # id2 = header["Header"].id2
        return cls(header, raw_data)

    def _parse(self):
        pass

    def as_dict(self):
        return {}

    def get_name(self):
        return (
            "Generic_" + chr(self.header["Header"].id1) + chr(self.header["Header"].id2)
        )
