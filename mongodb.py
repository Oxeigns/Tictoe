# mongodb.py
"""MongoDB helpers for TicTacToe bot (fixed + production-ready)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection

from config import Settings


def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


class Mongo:
    """Lightweight MongoDB helper."""

    def __init__(self, settings: Settings) -> None:
        self.client = MongoClient(settings.mongo_url, appname="TictoeBot")
        self.db = (
            self.client.get_default_database()
            if self.client.get_default_database()
            else self.client["tictactoe"]
        )
        self.groups: Collection = self.db["groups"]
        self.users: Collection = self.db["users"]
        self.games: Collection = self.db["games"]
        self.logs: Collection = self.db["logs"]

        # Indexes (important for speed + expiry scans)
        self.groups.create_index("activeGameId")
        self.groups.create_index("logChatId")

        self.users.create_index("username")
        self.users.create_index("lastSeen")

        self.games.create_index("groupId")
        self.games.create_index("status")
        self.games.create_index("joinExpiresAt")
        self.games.create_index("updatedAt")
        self.games.create_index([("groupId", 1), ("status", 1)])

        self.logs.create_index("ts")
        self.logs.create_index("groupId")
        self.logs.create_index("gameId")

        self.settings = settings

    # ---------------- Users ----------------

    def upsert_user(self, user: Dict[str, Any]) -> None:
        self.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "name": user.get("name"),
                    "username": user.get("username"),
                    "lastSeen": utcnow(),
                },
                "$setOnInsert": {
                    "stats": {"wins": 0, "losses": 0, "draws": 0, "gamesPlayed": 0},
                },
            },
            upsert=True,
        )

    def user_stats(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self.users.find_one({"_id": user_id})

    # ---------------- Groups ----------------

    def get_or_create_group(self, chat: Dict[str, Any]) -> Dict[str, Any]:
        doc = self.groups.find_one_and_update(
            {"_id": chat["id"]},
            {
                "$set": {"title": chat.get("title")},
                "$setOnInsert": {
                    "premiumEnabled": False,
                    "activeGameId": None,
                    "logChatId": None,
                    "settings": {
                        "joinTimeoutSec": getattr(self.settings, "join_timeout_sec", 60),
                        "turnTimeoutSec": getattr(self.settings, "turn_timeout_sec", 60),
                        "allowOpenJoin": True,
                    },
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return doc

    def set_log_group(self, group_id: int, log_chat_id: Optional[int]) -> None:
        self.groups.update_one({"_id": group_id}, {"$set": {"logChatId": log_chat_id}})

    def set_active_game_if_free(self, group_id: int, game_id: str) -> bool:
        # Ensure group doc exists (in case /startgame called first time)
        self.groups.update_one(
            {"_id": group_id},
            {"$setOnInsert": {"activeGameId": None, "premiumEnabled": False, "logChatId": None, "settings": {}}},
            upsert=True,
        )
        res = self.groups.find_one_and_update(
            {"_id": group_id, "activeGameId": None},
            {"$set": {"activeGameId": game_id}},
            return_document=ReturnDocument.AFTER,
        )
        return res is not None

    def clear_active_game_if_match(self, group_id: int, game_id: str) -> None:
        self.groups.update_one(
            {"_id": group_id, "activeGameId": game_id},
            {"$set": {"activeGameId": None}},
        )

    def all_group_ids(self) -> List[int]:
        return [doc["_id"] for doc in self.groups.find({}, {"_id": 1})]

    # ---------------- Games ----------------

    def create_game(self, game: Dict[str, Any]) -> Dict[str, Any]:
        self.games.insert_one(game)
        return game

    def fetch_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        return self.games.find_one({"_id": game_id})

    def update_game(self, game_id: str, update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.games.find_one_and_update(
            {"_id": game_id},
            update,
            return_document=ReturnDocument.AFTER,
        )

    def record_result(self, game_id: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.games.find_one_and_update(
            {"_id": game_id},
            {"$set": {"status": "ENDED", "result": result, "updatedAt": utcnow()}},
            return_document=ReturnDocument.AFTER,
        )

    def bump_stats(self, player_a: Optional[int], player_b: Optional[int], draw: bool = False) -> None:
        """
        Safe stats updater.

        - If draw=True: increments draws + gamesPlayed for both players (if present).
        - Else: player_a treated as winner, player_b treated as loser.
        """
        if draw:
            ids = [uid for uid in [player_a, player_b] if uid is not None]
            if ids:
                self.users.update_many(
                    {"_id": {"$in": ids}},
                    {"$inc": {"stats.draws": 1, "stats.gamesPlayed": 1}},
                )
            return

        # Non-draw: winner/loser
        if player_a is not None:
            self.users.update_one(
                {"_id": player_a},
                {"$inc": {"stats.wins": 1, "stats.gamesPlayed": 1}},
                upsert=True,
            )
        if player_b is not None:
            self.users.update_one(
                {"_id": player_b},
                {"$inc": {"stats.losses": 1, "stats.gamesPlayed": 1}},
                upsert=True,
            )

    # ---------------- Expiry helpers ----------------

    def list_expired_lobbies(self, now: dt.datetime) -> List[Dict[str, Any]]:
        """
        Returns LOBBY games whose joinExpiresAt has passed.
        """
        return list(
            self.games.find(
                {
                    "status": "LOBBY",
                    "joinExpiresAt": {"$ne": None, "$lte": now},
                },
                {"_id": 1, "groupId": 1, "status": 1, "joinExpiresAt": 1},
            ).limit(100)
        )

    def list_expired_turns(self, now: dt.datetime) -> List[Dict[str, Any]]:
        """
        Optional: Returns ACTIVE games where updatedAt older than turn_timeout_sec.
        Works if you set updatedAt on each move (your app.py does).
        """
        # If no turn timeout configured, return empty
        timeout_sec = getattr(self.settings, "turn_timeout_sec", 0) or 0
        if timeout_sec <= 0:
            return []

        cutoff = now - dt.timedelta(seconds=int(timeout_sec))
        return list(
            self.games.find(
                {
                    "status": "ACTIVE",
                    "updatedAt": {"$lte": cutoff},
                },
                {
                    "_id": 1,
                    "groupId": 1,
                    "status": 1,
                    "turn": 1,
                    "playerX": 1,
                    "playerO": 1,
                    "updatedAt": 1,
                },
            ).limit(100)
        )

    # ---------------- Stats views ----------------

    def group_stats(self, group_id: int) -> Dict[str, int]:
        """
        Simple stats summary based on games.result.type
        """
        cursor = self.games.find({"groupId": group_id}, {"result": 1})
        total = wins = draws = losses = 0
        for g in cursor:
            total += 1
            r = (g.get("result") or {}).get("type")
            if r in {"WIN", "FORFEIT", "TURN_TIMEOUT"}:
                wins += 1
            elif r == "DRAW":
                draws += 1
            elif r in {"CANCELLED", "ADMIN_END", "LOBBY_TIMEOUT", "INVITE_DECLINED"}:
                losses += 1  # treat as non-win outcomes for group summary
        return {"total": total, "wins": wins, "draws": draws, "losses": losses}

    # ---------------- Logs ----------------

    def log_event(self, payload: Dict[str, Any]) -> None:
        self.logs.insert_one({**payload, "ts": utcnow()})
