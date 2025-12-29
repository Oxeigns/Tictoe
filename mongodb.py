"""MongoDB helpers for TicTacToe bot."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection

from config import Settings


class Mongo:
    """Lightweight MongoDB helper."""

    def __init__(self, settings: Settings) -> None:
        self.client = MongoClient(settings.mongo_url, appname="TictoeBot")
        self.db = self.client.get_default_database() if self.client.get_default_database() else self.client["tictactoe"]
        self.groups: Collection = self.db["groups"]
        self.users: Collection = self.db["users"]
        self.games: Collection = self.db["games"]
        self.logs: Collection = self.db["logs"]

        self.groups.create_index("activeGameId")
        self.games.create_index("groupId")
        self.logs.create_index("ts")

    def upsert_user(self, user: Dict[str, Any]) -> None:
        self.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "name": user.get("name"),
                    "username": user.get("username"),
                    "lastSeen": dt.datetime.utcnow(),
                },
                "$setOnInsert": {
                    "stats": {"wins": 0, "losses": 0, "draws": 0, "gamesPlayed": 0}
                },
            },
            upsert=True,
        )

    def get_or_create_group(self, chat: Dict[str, Any]) -> Dict[str, Any]:
        doc = self.groups.find_one_and_update(
            {"_id": chat["id"]},
            {
                "$set": {"title": chat.get("title")},
                "$setOnInsert": {
                    "premiumEnabled": False,
                    "activeGameId": None,
                    "settings": {},
                    "logChatId": None,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return doc

    def set_active_game_if_free(self, group_id: int, game_id: str) -> bool:
        res = self.groups.find_one_and_update(
            {"_id": group_id, "activeGameId": None},
            {"$set": {"activeGameId": game_id}},
            return_document=ReturnDocument.AFTER,
        )
        return res is not None

    def clear_active_game_if_match(self, group_id: int, game_id: str) -> None:
        self.groups.update_one({"_id": group_id, "activeGameId": game_id}, {"$set": {"activeGameId": None}})

    def create_game(self, game: Dict[str, Any]) -> Dict[str, Any]:
        self.games.insert_one(game)
        return game

    def update_game(self, game_id: str, update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.games.find_one_and_update(
            {"_id": game_id},
            update,
            return_document=ReturnDocument.AFTER,
        )

    def fetch_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        return self.games.find_one({"_id": game_id})

    def append_move(self, game_id: str, move: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.games.find_one_and_update(
            {"_id": game_id},
            {"$push": {"moves": move}, "$set": {"updatedAt": dt.datetime.utcnow()}},
            return_document=ReturnDocument.AFTER,
        )

    def record_result(self, game_id: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.games.find_one_and_update(
            {"_id": game_id},
            {"$set": {"status": "ENDED", "result": result, "updatedAt": dt.datetime.utcnow()}},
            return_document=ReturnDocument.AFTER,
        )

    def bump_stats(self, winner: Optional[int], loser: Optional[int], draw: bool = False) -> None:
        if winner:
            self.users.update_one({"_id": winner}, {"$inc": {"stats.wins": 1, "stats.gamesPlayed": 1}})
        if loser:
            self.users.update_one({"_id": loser}, {"$inc": {"stats.losses": 1, "stats.gamesPlayed": 1}})
        if draw:
            self.users.update_many({"_id": {"$in": [winner, loser]}}, {"$inc": {"stats.draws": 1, "stats.gamesPlayed": 1}})

    def user_stats(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self.users.find_one({"_id": user_id})

    def group_stats(self, group_id: int) -> Dict[str, Any]:
        games = list(self.games.find({"groupId": group_id}))
        wins = draws = losses = 0
        for g in games:
            result = g.get("result") or {}
            if result.get("type") == "WIN":
                if result.get("winnerUserId") == g.get("playerX") or result.get("winnerUserId") == g.get("playerO"):
                    wins += 1
            elif result.get("type") == "DRAW":
                draws += 1
            elif result.get("type") == "FORFEIT":
                losses += 1
        return {"total": len(games), "wins": wins, "draws": draws, "losses": losses}

    def all_group_ids(self) -> list[int]:
        return [doc["_id"] for doc in self.groups.find({}, {"_id": 1})]

    def set_log_group(self, group_id: int, log_chat_id: Optional[int]) -> None:
        self.groups.update_one({"_id": group_id}, {"$set": {"logChatId": log_chat_id}})

    def log_event(self, payload: Dict[str, Any]) -> None:
        payload = {**payload, "ts": dt.datetime.utcnow()}
        self.logs.insert_one(payload)

