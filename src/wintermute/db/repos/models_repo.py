"""Repository for Model registry operations."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from wintermute.db.models import Model


class ModelRepo:
    """Model registry operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        version: str,
        weights_path: str,
        manifest_path: str,
        config: dict,
        metrics: dict | None = None,
        **kwargs,
    ) -> Model:
        """Register a new model version. Status = 'staged'.

        *metrics* dict may contain ``best_val_macro_f1``,
        ``best_val_accuracy``, ``best_val_auc_roc`` — these are
        extracted and set as top-level columns.

        *kwargs* must include ``vocab_size``, ``num_classes``, ``dims``
        (these are NOT NULL on the model).
        """
        # Extract metric columns from the metrics dict
        metric_fields = {}
        if metrics:
            for key in ("best_val_macro_f1", "best_val_accuracy", "best_val_auc_roc"):
                if key in metrics:
                    metric_fields[key] = metrics[key]

        # Filter kwargs to only valid Model attributes
        valid_attrs = {k: v for k, v in kwargs.items() if hasattr(Model, k)}

        model = Model(
            version=version,
            weights_path=weights_path,
            manifest_path=manifest_path,
            config=config,
            status="staged",
            **metric_fields,
            **valid_attrs,
        )
        self._session.add(model)
        self._session.flush()
        return model

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def promote(self, model_id: str) -> Model:
        """Promote to 'active'. Retires the currently active model of
        the same architecture.  Sets ``promoted_at``.

        Raises ``ValueError`` if *model_id* not found.
        """
        model = self._session.get(Model, model_id)
        if model is None:
            raise ValueError(f"Model not found: {model_id}")

        # Retire ALL currently active models of same architecture
        stmt = select(Model).where(
            Model.architecture == model.architecture,
            Model.status == "active",
            Model.id != model_id,
        )
        for active_model in self._session.execute(stmt).scalars().all():
            active_model.status = "retired"
            active_model.retired_at = datetime.now(timezone.utc)

        model.status = "active"
        model.promoted_at = datetime.now(timezone.utc)
        self._session.flush()
        return model

    def retire(self, model_id: str) -> None:
        """Set status to 'retired', set ``retired_at``.

        Raises ``ValueError`` if not found.
        """
        model = self._session.get(Model, model_id)
        if model is None:
            raise ValueError(f"Model not found: {model_id}")

        model.status = "retired"
        model.retired_at = datetime.now(timezone.utc)
        self._session.flush()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def active(self, architecture: str = "WintermuteMalwareDetector") -> Model | None:
        """Get the currently active model for a given architecture."""
        stmt = (
            select(Model).where(Model.status == "active").where(Model.architecture == architecture)
        )
        return self._session.execute(stmt).scalars().first()

    def history(self, limit: int = 20) -> list[Model]:
        """All model versions, newest first (``created_at DESC``)."""
        stmt = select(Model).order_by(Model.created_at.desc()).limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def compare(self, model_a_id: str, model_b_id: str) -> dict:
        """Compare two models by metrics.

        Returns dict with ``model_a`` and ``model_b`` keys, each
        containing id, version, status, and metric fields.
        """
        a = self._session.get(Model, model_a_id)
        b = self._session.get(Model, model_b_id)

        if a is None:
            raise ValueError(f"Model not found: {model_a_id}")
        if b is None:
            raise ValueError(f"Model not found: {model_b_id}")

        def _snapshot(m: Model) -> dict:
            return {
                "id": m.id,
                "version": m.version,
                "status": m.status,
                "best_val_macro_f1": m.best_val_macro_f1,
                "best_val_accuracy": m.best_val_accuracy,
                "best_val_auc_roc": m.best_val_auc_roc,
            }

        return {"model_a": _snapshot(a), "model_b": _snapshot(b)}
