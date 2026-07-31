"""EntropyDecayCallback의 decay_fraction 동작 검증.

decay_fraction=1.0(기본)이면 학습 전체 구간에 걸쳐 start->end로 선형
감쇠하고, 작은 값(예: 0.2)이면 그 구간(처음 20%)에서 이미 end에 도달해
그 뒤로는 end로 고정돼야 한다.
"""
from types import SimpleNamespace

from agent.train_progress import EntropyDecayCallback


def _cb(start, end, decay_fraction):
    cb = EntropyDecayCallback(start, end, decay_fraction=decay_fraction)
    cb.model = SimpleNamespace(ent_coef=None, _current_progress_remaining=1.0)
    return cb


def _set_progress(cb, elapsed_fraction: float):
    cb.model._current_progress_remaining = 1.0 - elapsed_fraction


def test_full_decay_fraction_linear_over_whole_run():
    cb = _cb(start=0.05, end=0.0, decay_fraction=1.0)

    _set_progress(cb, 0.0)
    cb._on_step()
    assert cb.model.ent_coef == 0.05

    _set_progress(cb, 0.5)
    cb._on_step()
    assert abs(cb.model.ent_coef - 0.025) < 1e-9

    _set_progress(cb, 1.0)
    cb._on_step()
    assert abs(cb.model.ent_coef - 0.0) < 1e-9


def test_short_decay_fraction_reaches_end_early_and_stays():
    cb = _cb(start=0.05, end=0.0, decay_fraction=0.2)

    _set_progress(cb, 0.0)
    cb._on_step()
    assert cb.model.ent_coef == 0.05

    _set_progress(cb, 0.1)  # 0.2 구간의 절반 지점
    cb._on_step()
    assert abs(cb.model.ent_coef - 0.025) < 1e-9

    _set_progress(cb, 0.2)  # 감쇠 구간 끝 지점
    cb._on_step()
    assert abs(cb.model.ent_coef - 0.0) < 1e-9

    # 감쇠 구간을 지난 뒤(예: 80% 지점)에는 end에 고정돼야 한다
    _set_progress(cb, 0.8)
    cb._on_step()
    assert abs(cb.model.ent_coef - 0.0) < 1e-9
