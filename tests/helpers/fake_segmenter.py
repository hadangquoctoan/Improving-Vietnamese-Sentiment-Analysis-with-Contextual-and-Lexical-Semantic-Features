class FakeSegmenter:
    def __init__(self, mapping=None):
        self.mapping = dict(mapping or {})
        self.calls = 0

    def segment(self, text):
        self.calls += 1
        return self.mapping.get(text, text)


FIXTURE_MAPPING = {
    "hào hứng": "hào_hứng",
    "khô cằn": "khô_cằn",
    "khô hạn": "khô_hạn",
    "thái quá": "thái_quá",
    "giỏi": "giỏi",
}
