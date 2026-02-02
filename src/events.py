# src/events.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, List, Optional


@dataclass
class ZoneIDState:
    inside: bool = False
    enter_t: Optional[float] = None
    loiter_fired: bool = False
    last_t: float = 0.0


@dataclass
class Event:
    t: float
    type: str            # "enter" | "exit" | "loiter"
    zone: str
    obj_id: int
    meta: Dict


class EventEngine:
    def __init__(
        self,
        loiter_threshold_s: float = 10.0,
        exit_debounce_s: float = 0.0,
    ):
        self.loiter_threshold_s = loiter_threshold_s
        self.exit_debounce_s = exit_debounce_s

        # state[(zone_name, obj_id)] -> ZoneIDState
        self.state: Dict[tuple[str, int], ZoneIDState] = {}

    def update(
        self,
        zone_hits: Dict[int, Set[str]],
        t: float,
    ) -> List[Event]:
        """
        zone_hits: {id: {"zone_a", ...}}
        t: timestamp (seconds, monotonic)
        """
        events: List[Event] = []

        # Build reverse mapping: zone -> set(ids inside)
        zones_now: Dict[str, Set[int]] = {}
        for obj_id, zones in zone_hits.items():
            for z in zones:
                zones_now.setdefault(z, set()).add(obj_id)

        # Collect all known (zone, id) pairs
        known_pairs = set(self.state.keys())
        current_pairs = set()

        for obj_id, zones in zone_hits.items():
            for z in zones:
                current_pairs.add((z, obj_id))

        # Handle ENTER / LOITER
        for (z, obj_id) in current_pairs:
            key = (z, obj_id)
            st = self.state.get(key)

            if st is None:
                # First time seeing this ID in this zone
                st = ZoneIDState(
                    inside=True,
                    enter_t=t,
                    loiter_fired=False,
                    last_t=t,
                )
                self.state[key] = st

                events.append(Event(
                    t=t,
                    type="enter",
                    zone=z,
                    obj_id=obj_id,
                    meta={},
                ))
                continue

            # Was already known
            if not st.inside:
                # Outside -> inside transition
                st.inside = True
                st.enter_t = t
                st.loiter_fired = False
                st.last_t = t

                events.append(Event(
                    t=t,
                    type="enter",
                    zone=z,
                    obj_id=obj_id,
                    meta={},
                ))
            else:
                # Still inside
                st.last_t = t
                if (
                    not st.loiter_fired
                    and st.enter_t is not None
                    and (t - st.enter_t) >= self.loiter_threshold_s
                ):
                    st.loiter_fired = True
                    events.append(Event(
                        t=t,
                        type="loiter",
                        zone=z,
                        obj_id=obj_id,
                        meta={"duration_s": t - st.enter_t},
                    ))

        # Handle EXIT
        for (z, obj_id) in list(known_pairs):
            key = (z, obj_id)
            if key in current_pairs:
                continue

            st = self.state[key]
            if not st.inside:
                continue

            # Outside this frame
            if (t - st.last_t) >= self.exit_debounce_s:
                duration = None
                if st.enter_t is not None:
                    duration = t - st.enter_t

                st.inside = False
                st.enter_t = None
                st.loiter_fired = False

                events.append(Event(
                    t=t,
                    type="exit",
                    zone=z,
                    obj_id=obj_id,
                    meta={"duration_s": duration},
                ))

        return events


# ---- self-test ----
def _self_test() -> None:
    engine = EventEngine(loiter_threshold_s=3.0)

    timeline = [
        # t, zone_hits
        (0.0, {1: set()}),
        (1.0, {1: {"zone_a"}}),   # enter
        (2.0, {1: {"zone_a"}}),
        (4.2, {1: {"zone_a"}}),   # loiter
        (5.0, {1: set()}),        # exit
    ]

    all_events: List[Event] = []

    for t, hits in timeline:
        evs = engine.update(hits, t)
        for e in evs:
            print(e)
        all_events.extend(evs)

    types = [e.type for e in all_events]
    assert types == ["enter", "loiter", "exit"], f"Unexpected event order: {types}"
    print("events.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
