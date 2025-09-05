# ...existing code (MGAModel definition)...
from mga_yolo.model.trainer import MGATrainer  # noqa: E402
from mga_yolo.model.predictor import MGAPredictor  # noqa: E402
from mga_yolo.model.validator import MGAValidator  # noqa: E402
from mga_yolo.model.model import MGAModel  # noqa: E402

__all__ = ["MGAModel", "MGATrainer", "MGAPredictor", "MGAValidator"]


# at the end of __init__ or before training
import sys
dups = [m for m in sys.modules if m.startswith("ultralytics")]
if len(set(dups)) > 1:
    from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER
    LOGGER.info(f"[MGA] Multiple ultralytics modules loaded: {dups}. "
                f"This can cause pickling class-identity errors.")
