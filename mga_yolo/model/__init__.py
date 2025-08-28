# ...existing code (MGAModel definition)...
from mga_yolo.model.trainer import MGATrainer  # noqa: E402
from mga_yolo.model.predictor import MGAPredictor  # noqa: E402
from mga_yolo.model.validator import MGAValidator  # noqa: E402
from mga_yolo.model.model import MGAModel  # noqa: E402

__all__ = ["MGAModel", "MGATrainer", "MGAPredictor", "MGAValidator"]