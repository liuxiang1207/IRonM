"""按天生成模拟用户的记录与隐藏真值。

每个随机值只由（种子、用户、日期、项目）决定（按键哈希出随机数），
因此增减一个陷阱只影响它自己那天，其余天的记录不变。
"""

import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from ironm.contracts.events import EVENT_ADAPTER, Event
from sim.persona import Persona, Phase, Trap

_SLOTS = (
    ("breakfast", time(8, 0), 0.25),
    ("lunch", time(12, 30), 0.40),
    ("dinner", time(19, 0), 0.35),
)


@dataclass(frozen=True)
class DayEvents:
    index: int  # 第几天，从 1 开始
    date: date
    closes_at: datetime  # 当天结束、记录全部提交的模拟时刻（UTC）
    events: tuple[Event, ...]


@dataclass(frozen=True)
class TrapRecord:
    kind: str
    day: int | None
    event_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True)
class Truth:
    true_weight_kg: tuple[float, ...]  # 每天的真实体重
    traps: tuple[TrapRecord, ...]


@dataclass(frozen=True)
class Simulation:
    persona_id: str
    days: tuple[DayEvents, ...]
    truth: Truth


def _rng(p: Persona, *keys: object) -> random.Random:
    digest = hashlib.sha256(json.dumps([p.seed, p.id, *keys]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8]))


def _uuid(p: Persona, *keys: object) -> str:
    return str(uuid.UUID(int=_rng(p, "id", *keys).getrandbits(128), version=4))


def _phase[T: Phase](phases: tuple[T, ...], week: int) -> T:
    current = phases[0]
    for ph in phases:
        if ph.from_week <= week:
            current = ph
    return current


def _true_weights(p: Persona) -> tuple[float, ...]:
    days = p.weeks * 7
    weights = [p.profile.start_weight_kg]
    for d in range(2, days + 1):
        week = (d - 2) // 7 + 1
        weights.append(weights[-1] + _phase(p.hidden.weight_phases, week).kg_per_week / 7)
    return tuple(weights)


def _clamp(v: int, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, v))


class _DayBuilder:
    def __init__(self, p: Persona, index: int, day: date) -> None:
        self.p, self.index, self.day = p, index, day
        self.events: list[Event] = []

    def add(self, kind: str, at: time, payload: dict[str, object], **extra: object) -> Event:
        p = self.p
        occurred = datetime.combine(self.day, at, p.tz())
        data = {
            "event_id": _uuid(p, self.index, kind, len(self.events)),
            "user_id": p.user_id,
            "type": kind,
            "occurred_at": occurred.isoformat(),
            "recorded_at": (occurred + timedelta(minutes=2)).isoformat(),
            "source": "user",
            "confirmed": True,
            "payload": payload,
            **extra,
        }
        event = EVENT_ADAPTER.validate_json(json.dumps(data))
        self.events.append(event)
        return event


def _silent(traps: tuple[Trap, ...], index: int) -> bool:
    for t in traps:
        if t.kind == "stop_logging" and t.from_day is not None and index >= t.from_day:
            return True
        in_gap = t.from_day is not None and t.to_day is not None and t.from_day <= index <= t.to_day
        if t.kind == "gap" and in_gap:
            return True
    return False


def generate(p: Persona) -> Simulation:
    truth = _true_weights(p)
    traps_by_day = {t.day: t for t in p.traps if t.day is not None}
    switch = next((t for t in p.traps if t.kind == "device_switch"), None)
    trap_events: dict[int, list[uuid.UUID]] = {i: [] for i in range(len(p.traps))}
    days: list[DayEvents] = []

    for index in range(1, p.weeks * 7 + 1):
        day = p.start_date + timedelta(days=index - 1)
        week = (index - 1) // 7 + 1
        weekday = day.strftime("%a").lower()
        b = _DayBuilder(p, index, day)
        closes = datetime.combine(day, time(23, 0), p.tz())

        if not _silent(p.traps, index):
            # 称重
            trap = traps_by_day.get(index)
            weigh_today = weekday in p.habits.weigh_days or (
                trap is not None and trap.kind in {"spike", "unit_error"}
            )
            if weigh_today:
                true_w = truth[index - 1]
                measured = true_w + _rng(p, "weight", index).gauss(0, p.hidden.weight_noise_sd_kg)
                stream = p.habits.scale_stream
                switched = switch is not None and switch.day is not None and index >= switch.day
                if switched and switch is not None:
                    stream = str(switch.stream)
                    measured += float(switch.offset_kg or 0)
                raw = None
                if trap is not None and trap.kind == "spike":
                    measured = true_w + float(trap.delta_kg or 0)
                if trap is not None and trap.kind == "unit_error":
                    measured = measured * 2  # 把“斤”当“公斤”记
                    raw = f"称了 {measured:.1f}"
                e = b.add(
                    "weight",
                    time(7, 30),
                    {"weight_kg": round(measured, 1)},
                    stream_key=stream,
                    raw=raw,
                )
                for i, t in enumerate(p.traps):
                    if (t.day == index and t.kind in {"spike", "unit_error"}) or (
                        t.kind == "device_switch" and switched
                    ):
                        trap_events[i].append(e.event_id)

            # 饮食
            if p.habits.meal_logging != "none":
                for slot, at, share in _SLOTS:
                    payload: dict[str, object] = {"slot": slot}
                    if p.habits.meal_logging == "kcal":
                        noise = _rng(p, "meal", index, slot).gauss(1, 0.12)
                        payload["kcal"] = round(p.hidden.daily_kcal * share * noise)
                    b.add("meal", at, payload)

            # 训练
            if weekday in p.habits.training_days:
                u = _rng(p, "workout", index).random()
                p_done = _phase(p.hidden.adherence_phases, week).p_done
                status = "done" if u < p_done else ("half" if u < (1 + p_done) / 2 else "skipped")
                payload = {"status": status}
                if status != "skipped":
                    level = _phase(p.hidden.fatigue_phases, week).level
                    payload["effort"] = _clamp(level + _rng(p, "effort", index).randint(-1, 1))
                b.add("workout_check", time(20, 0), payload)

            # 每周签到
            if weekday == p.habits.checkin_day:
                r = _rng(p, "checkin", index)
                level = _phase(p.hidden.fatigue_phases, week).level
                fatigue = _clamp(level + r.randint(-1, 1))
                weekly_change = truth[index - 1] - truth[max(0, index - 8)]
                fit = (
                    "looser"
                    if weekly_change < -0.2
                    else "tighter"
                    if weekly_change > 0.2
                    else "same"
                )
                p_done = _phase(p.hidden.adherence_phases, week).p_done
                b.add(
                    "checkin",
                    time(21, 30),
                    {
                        "fatigue": fatigue,
                        "sleep_quality": _clamp(6 - fatigue + r.randint(-1, 0)),
                        "diet_adherence_days": max(0, min(7, round(7 * p_done + r.uniform(-1, 1)))),
                        "clothes_fit": fit,
                    },
                )

        days.append(DayEvents(index, day, closes.astimezone(UTC), tuple(b.events)))

    records = tuple(
        TrapRecord(t.kind, t.day if t.day is not None else t.from_day, tuple(trap_events[i]))
        for i, t in enumerate(p.traps)
    )
    return Simulation(p.id, tuple(days), Truth(truth, records))
