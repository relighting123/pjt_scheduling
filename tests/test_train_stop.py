"""학습 중지 요청 상태 테스트."""
from agent.train_progress import TrainProgressState, StopTrainingCallback


def test_train_progress_stop_flow():
    state = TrainProgressState()
    state.set_running(total_timesteps=1000)
    assert not state.is_stop_requested()

    state.request_stop()
    cb = StopTrainingCallback(state)
    assert cb._on_step() is False

    state.set_stopped()
    assert state.status == "stopped"
    assert not state.is_stop_requested()


def test_stop_training_requires_active_thread():
    from api.train_service import stop_training, is_training

    assert not is_training()
    assert stop_training() is False
