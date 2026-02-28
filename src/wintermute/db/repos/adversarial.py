"""Repository for adversarial vault and cycle tracking."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from wintermute.db.models import AdversarialCycle, AdversarialVariant, Sample


class AdversarialRepo:
    """Adversarial vault and cycle tracking."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Cycle management
    # ------------------------------------------------------------------

    def start_cycle(
        self,
        defender_model_id: str | None = None,
        cycle_number: int | None = None,
    ) -> AdversarialCycle:
        """Begin a new adversarial cycle. Sets ``started_at``."""
        cycle = AdversarialCycle(
            defender_model_id=defender_model_id,
            cycle_number=cycle_number,
        )
        self._session.add(cycle)
        self._session.flush()
        return cycle

    def complete_cycle(self, cycle_id: str, stats: dict) -> None:
        """Record cycle completion. Sets ``completed_at`` and stats fields.

        *stats* may contain: ``episodes_played``, ``total_evasions``,
        ``evasion_rate``, ``mean_confidence_drop``, ``vault_samples_used``,
        ``defender_f1_before``, ``defender_f1_after``.

        Sets ``retrained=True`` if ``defender_f1_after`` is present.
        """
        cycle = self._session.get(AdversarialCycle, cycle_id)
        if cycle is None:
            raise ValueError(f"Adversarial cycle not found: {cycle_id}")

        cycle.completed_at = datetime.now(timezone.utc)

        for key in (
            "episodes_played",
            "total_evasions",
            "evasion_rate",
            "mean_confidence_drop",
            "vault_samples_used",
            "defender_f1_before",
            "defender_f1_after",
        ):
            if key in stats:
                setattr(cycle, key, stats[key])

        if "defender_f1_after" in stats:
            cycle.retrained = True

        self._session.flush()

    # ------------------------------------------------------------------
    # Variant CRUD
    # ------------------------------------------------------------------

    def store_variant(
        self,
        parent_sha256: str,
        cycle_id: str,
        mutated_tokens: list[int],
        mutations: list[dict],
        confidence_before: float,
        confidence_after: float,
        **kwargs,
    ) -> AdversarialVariant:
        """Store an adversarial variant. Auto-computes:

        - ``mutation_count`` = len(mutations)
        - ``confidence_delta`` = confidence_after - confidence_before
        - ``achieved_evasion`` = confidence_after < 0.5
        - ``modification_pct`` from kwargs
        """
        mutation_count = len(mutations)
        confidence_delta = confidence_after - confidence_before
        achieved_evasion = confidence_after < 0.5
        modification_pct = kwargs.pop("modification_pct", 0.0)

        # Filter remaining kwargs to valid attributes
        valid_attrs = {k: v for k, v in kwargs.items() if hasattr(AdversarialVariant, k)}

        variant = AdversarialVariant(
            parent_sha256=parent_sha256,
            cycle_id=cycle_id,
            mutated_token_ids=mutated_tokens,
            mutations_applied=mutations,
            mutation_count=mutation_count,
            modification_pct=modification_pct,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            confidence_delta=confidence_delta,
            achieved_evasion=achieved_evasion,
            **valid_attrs,
        )
        self._session.add(variant)
        self._session.flush()
        return variant

    # ------------------------------------------------------------------
    # Vault queries
    # ------------------------------------------------------------------

    def get_vault(
        self,
        limit: int = 1000,
        evasion_only: bool = True,
        unused_only: bool = True,
    ) -> list[AdversarialVariant]:
        """Retrieve vault samples for retraining.

        Default: evasive + unused.
        Order by ``confidence_delta ASC`` (biggest drops first).
        """
        stmt = select(AdversarialVariant)

        if evasion_only:
            stmt = stmt.where(AdversarialVariant.achieved_evasion.is_(True))
        if unused_only:
            stmt = stmt.where(AdversarialVariant.used_in_retraining.is_(False))

        stmt = stmt.order_by(AdversarialVariant.confidence_delta.asc()).limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def mark_retrained(self, variant_ids: list[str], training_run_id: str) -> int:
        """Mark variants as used in retraining.

        Sets ``used_in_retraining=True`` and ``retraining_run_id``.
        Returns count updated.
        """
        if not variant_ids:
            return 0

        stmt = select(AdversarialVariant).where(AdversarialVariant.id.in_(variant_ids))
        variants = list(self._session.execute(stmt).scalars().all())

        for v in variants:
            v.used_in_retraining = True
            v.retraining_run_id = training_run_id

        self._session.flush()
        return len(variants)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def vulnerability_report(self) -> list[dict]:
        """Per-family evasion stats.

        JOINs ``adversarial_variants`` -> ``samples`` ON
        ``parent_sha256``.

        Returns::

            [
                {
                    "family": str,
                    "total_attacks": int,
                    "evasions": int,
                    "evasion_rate": float,
                    "mean_delta": float,
                },
                ...
            ]
        """
        stmt = (
            select(
                Sample.family,
                func.count().label("total_attacks"),
                func.sum(
                    case(
                        (AdversarialVariant.achieved_evasion.is_(True), 1),
                        else_=0,
                    )
                ).label("evasions"),
                func.avg(AdversarialVariant.confidence_delta).label("mean_delta"),
            )
            .join(
                Sample,
                AdversarialVariant.parent_sha256 == Sample.sha256,
            )
            .group_by(Sample.family)
        )

        rows = self._session.execute(stmt).all()
        result = []
        for row in rows:
            total = row.total_attacks
            evasions = int(row.evasions)
            result.append(
                {
                    "family": row.family,
                    "total_attacks": total,
                    "evasions": evasions,
                    "evasion_rate": evasions / total if total > 0 else 0.0,
                    "mean_delta": float(row.mean_delta) if row.mean_delta is not None else 0.0,
                }
            )
        return result
