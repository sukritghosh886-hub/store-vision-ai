from .real_event_adapter import build_features
from .predict import predict_risk


def analyze_visit(visit_id):
    """
    Run Store Vision ML analysis for a completed visit.

    The ML layer is optional and must not break
    the existing Store Vision workflow.
    """

    features = build_features(visit_id)

    result = predict_risk(features)

    return {
        "visit_id": visit_id,
        "ml_analysis": result,
    }