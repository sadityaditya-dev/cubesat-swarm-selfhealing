"""Self-healing role manager and bounded HCW feedback controller."""
from __future__ import annotations
import math
import numpy as np
from ..models import Satellite
from .assignment import RoleAssigner, ROLE_BY_NAME, CRITICAL_ROLES

class SelfHealingController:
    def __init__(self, satellites: dict[str, Satellite], mean_motion_rad_s: float,
                 kp: float = 2.0e-6, kd: float = 2.0e-3, safety_distance_m: float = 250.0,
                 coverage_evaluator=None, communication_range_m: float = 18_000.0,
                 packet_loss_probability: float = 0.0):
        self.satellites = satellites
        self.n = float(mean_motion_rad_s)
        self.kp, self.kd = float(kp), float(kd)
        self.safety_distance_m = float(safety_distance_m)
        self.coverage_evaluator = coverage_evaluator
        self.communication_range_m = float(communication_range_m)
        self.packet_loss_probability = float(packet_loss_probability)
        self.isolated_ids: set[str] = set()
        self.assigner = RoleAssigner()
        self.assignment: dict[str, str] = {}
        self.assignment_log: list[dict[str, object]] = []
        self.degraded = False
        self._membership: tuple[tuple[str, tuple[str, ...]], ...] = ()
        self.reconcile(0.0, force=True)

    def reconcile(self, time_s: float, force: bool = False) -> None:
        active_ids = tuple(sorted(sid for sid, s in self.satellites.items() if s.is_operational))
        # Capability fingerprint matters as well as membership: a live spacecraft can
        # lose a sensor payload and invalidate its current role without losing its bus.
        membership_signature = tuple(sorted(
            (sid, tuple(sorted(s.effective_capabilities)))
            for sid, s in self.satellites.items() if s.is_operational
        ))
        if not force and membership_signature == self._membership:
            return
        previous = dict(self.assignment)
        coverage_contributions = {}
        if self.coverage_evaluator is not None:
            coverage_contributions = self.coverage_evaluator.individual_contributions(self.satellites)
        result = self.assigner.assign(
            self.satellites.values(), previous,
            communication_range_m=self.communication_range_m,
            packet_loss_probability=self.packet_loss_probability,
            coverage_contributions=coverage_contributions,
            isolated_ids=self.isolated_ids,
        )
        self._membership = membership_signature
        if result.feasible:
            self.assignment = result.assignment
            self.degraded = False
            for sat in self.satellites.values():
                sat.current_role = self.assignment.get(sat.satellite_id, "RESERVE" if sat.is_operational else sat.current_role)
            event = {"time_s": float(time_s), "type": "role_reassignment", "feasible": True,
                     "assignment": dict(self.assignment), "active_ids": list(active_ids), "cost": result.cost,
                     "reason": result.reason}
        else:
            self.degraded = True
            # Preserve only currently active, capability-valid parts of the prior
            # assignment. Failed spacecraft must never remain assigned in a degraded
            # result, and roles made infeasible by a payload fault are dropped.
            filtered: dict[str, str] = {}
            for sid, role_name in previous.items():
                sat = self.satellites.get(sid)
                role = ROLE_BY_NAME.get(role_name)
                if sat is not None and sat.is_operational and role is not None and sat.effective_capabilities & role.required_any:
                    filtered[sid] = role_name
            self.assignment = filtered
            for sat in self.satellites.values():
                if not sat.is_operational:
                    continue
                sat.current_role = self.assignment.get(sat.satellite_id, "UNASSIGNED")
            event = {"time_s": float(time_s), "type": "assignment_infeasible", "feasible": False,
                     "assignment": dict(self.assignment), "active_ids": list(active_ids),
                     "cost": None, "reason": result.reason}
        self.assignment_log.append(event)

    def control(self, sat: Satellite, time_s: float) -> tuple[float, float]:
        self.reconcile(time_s)
        if not sat.is_operational or self.degraded:
            return (0.0, 0.0)
        role_name = self.assignment.get(sat.satellite_id)
        if role_name is None:
            return (0.0, 0.0)
        role = ROLE_BY_NAME[role_name]
        x, y, vx, vy = map(float, sat.state)
        # Target is stationary in LVLH. Cancel HCW coupling terms and apply bounded PD feedback.
        ex, ey = x, y - role.target_y_m
        ux = -3.0 * self.n * self.n * x - 2.0 * self.n * vy - self.kp * ex - self.kd * vx
        uy = 2.0 * self.n * vx - self.kp * ey - self.kd * vy
        # Conservative collision-avoidance nudge when a close neighbour is detected.
        for other in self.satellites.values():
            if other.satellite_id == sat.satellite_id or not other.is_operational:
                continue
            delta = sat.state[:2] - other.state[:2]
            distance = float(np.linalg.norm(delta))
            if 1e-6 < distance < self.safety_distance_m:
                direction = delta / distance
                ux += 2.0e-4 * direction[0]
                uy += 2.0e-4 * direction[1]
        return (ux, uy)
