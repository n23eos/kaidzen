import pytest


class FakeLLM:
    """Отдаёт заранее заданные объекты, записывает вызовы. Ничего не шлёт в сеть."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def candidate(tmp_path):
    """Валидный кандидат на диске, собранный хелпером из test_candidate."""
    from tests.test_candidate import make_candidate
    from kaidzen.candidate import load_candidate
    return load_candidate(make_candidate(tmp_path))
