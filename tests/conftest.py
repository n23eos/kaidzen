import pytest

from kaidzen.candidate import REQUIRED_ROLES
from kaidzen.state import ApiUsage


class FakeLLM:
    """Фальшивый бэкенд: отдаёт заготовленные объекты, записывает вызовы.

    Ничего не шлёт в сеть. usage объявлен, как у настоящего бэкенда, — на нём
    держится проверка суммирования расхода по нескольким бэкендам.
    """

    supports_web_search = True

    def __init__(self, outputs, usage: ApiUsage | None = None):
        self.outputs = list(outputs)
        self.calls = []
        self.usage = usage if usage is not None else ApiUsage()

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def as_backends(backend, **per_role) -> dict:
    """Отображение «роль → бэкенд» для run_pipeline.

    По умолчанию все роли идут на один фальшивый бэкенд (обычный случай:
    весь цикл на подписке); per_role подменяет бэкенд отдельным ролям.
    """
    return {role: per_role.get(role, backend) for role in REQUIRED_ROLES}


@pytest.fixture
def candidate(tmp_path):
    """Валидный кандидат на диске, собранный хелпером из test_candidate."""
    from tests.test_candidate import make_candidate
    from kaidzen.candidate import load_candidate
    return load_candidate(make_candidate(tmp_path))
