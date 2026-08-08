import inspect

from model import Settings, build_pipeline


def test_core_is_service_ready() -> None:
    pipeline = build_pipeline(Settings())
    assert callable(pipeline.run)
    assert callable(pipeline.analyze_one)
    signature = inspect.signature(pipeline.analyze_one)
    assert set(signature.parameters) == {"task", "context"}

