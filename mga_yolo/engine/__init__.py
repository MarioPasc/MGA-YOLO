# ...existing code (MGAModel definition)...
from mga_yolo.engine.train import MGATrainer  # noqa: E402
from mga_yolo.engine.predict import MGAPredictor  # noqa: E402
from mga_yolo.engine.val import MGAValidator  # noqa: E402
from mga_yolo.engine.model import MGAModel  # noqa: E402

__all__ = ["MGAModel", "MGATrainer", "MGAPredictor", "MGAValidator"]