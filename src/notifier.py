# src/notifier.py
from __future__ import annotations

import os
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class Notification:
    t: float
    message: str
    zone: str
    obj_id: int


class Notifier:
    """
    Pushover-backed notifier.
    Sends real push notifications to phone.
    """

    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

    def __init__(self, cooldown_s: float = 300.0):
        self.cooldown_s = cooldown_s
        self._last_sent_t: Optional[float] = None

        self.user_key = os.getenv("PUSHOVER_USER_KEY")
        self.api_token = os.getenv("PUSHOVER_API_TOKEN")

        if not self.user_key or not self.api_token:
            raise RuntimeError(
                "Pushover credentials not found. "
                "Set PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN."
            )

    def should_send(self, t: float) -> bool:
        if self._last_sent_t is None:
            return True
        return (t - self._last_sent_t) >= self.cooldown_s

    def send(self, notification: Notification) -> None:
        if not self.should_send(notification.t):
            return

        payload = {
            "token": self.api_token,
            "user": self.user_key,
            "title": "Autonomous Vision Node",
            "message": notification.message,
            "priority": 0,
        }

        try:
            r = requests.post(
                self.PUSHOVER_URL,
                data=payload,
                timeout=5,
            )
            r.raise_for_status()
            self._last_sent_t = notification.t
            print("[NOTIFY] Push notification sent")

        except Exception as e:
            print(f"[NOTIFY ERROR] {e}")

