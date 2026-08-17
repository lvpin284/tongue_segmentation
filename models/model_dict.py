from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_tongueseg.build_tonguesegsam import tonguesegsam_model_registry


def _normalize_model_name(modelname):
    if modelname is None:
        return ""
    return str(modelname).strip().lower()


def get_model(modelname="TongueSegSAM", args=None, opt=None):
    normalized_name = _normalize_model_name(modelname)
    sam_ckpt = getattr(args, "sam_ckpt", None)

    if normalized_name in {"sam"}:
        return sam_model_registry["vit_b"](checkpoint=sam_ckpt)

    # Keep legacy aliases for backward compatibility, but expose backend-agnostic names.
    if normalized_name in {"tonguesegsam", "tongue_sam", "tongueseg-sam"}:
        return tonguesegsam_model_registry["vit_b"](args=args, checkpoint=sam_ckpt)

    raise RuntimeError("Could not find the model:", modelname)
