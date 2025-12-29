# app.py
"""Telegram TicTacToe bot using python-telegram-bot v20 (fixed + upgraded)."""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from cachetools import TTLCache
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatType, ParseMode
from telegram.error import RetryAfter, TimedOut
from telegram.ext import (
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from config import is_owner, load_settings
from mongodb import Mongo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ------------------------- Helpers -----------------------------------------

def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


def build_card(title: str, body: List[str], footer: Optional[str] = None) -> str:
    lines = [f"<b>{title}</b>", "──────────────────"]
    lines.extend(body)
    if footer:
        lines.append("──────────────────")
        lines.append(footer)
    return "\n".join(lines)


def winner_lines() -> List[Tuple[int, int, int]]:
    return [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]


def check_winner(board: List[str]) -> Tuple[Optional[str], Optional[List[int]]]:
    for a, b, c in winner_lines():
        if board[a] and board[a] == board[b] == board[c]:
            return board[a], [a, b, c]
    if all(board):
        return "DRAW", None
    return None, None


def mention_html(user_id: int, name: str) -> str:
    # Avoid importing helpers; keep it simple
    return f'<a href="tg://user?id={user_id}">{name}</a>'


# ------------------------- Bot ---------------------------------------------

class TicToeBot:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.mongo = Mongo(settings)

        # Callback spam throttle (per-user)
        self.throttle_cache: TTLCache[int, float] = TTLCache(
            maxsize=2000, ttl=settings.callback_throttle_sec
        )

        # Background expiry task
        self._expiry_task: Optional[asyncio.Task] = None

    # --------------------- Throttle ---------------------

    def _throttle(self, user_id: int) -> bool:
        if user_id in self.throttle_cache:
            return True
        self.throttle_cache[user_id] = utcnow().timestamp()
        return False

    # --------------------- DB helpers ---------------------

    def current_game(self, group_id: int) -> Optional[Dict]:
        group = self.mongo.groups.find_one({"_id": group_id})
        if not group:
            return None
        game_id = group.get("activeGameId")
        if not game_id:
            return None
        return self.mongo.fetch_game(game_id)

    # --------------------- UI builders ---------------------

    def lobby_keyboard(self, game_id: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Join Game", callback_data=f"ttt:join:{game_id}")],
                [InlineKeyboardButton("❌ Cancel Lobby", callback_data=f"ttt:cancel:{game_id}")],
            ]
        )

    def invite_keyboard(self, game_id: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Accept", callback_data=f"ttt:accept:{game_id}")],
                [InlineKeyboardButton("❌ Decline", callback_data=f"ttt:decline:{game_id}")],
            ]
        )

    def sign_keyboard(self, game_id: str, locked: Optional[str] = None) -> InlineKeyboardMarkup:
        # locked: "x" or "o" to show disabled feel
        if locked == "x":
            row = [
                InlineKeyboardButton("✅ ❌ X Selected", callback_data=f"ttt:noop:{game_id}"),
                InlineKeyboardButton("0️⃣ Choose O", callback_data=f"ttt:sign:o:{game_id}"),
            ]
        elif locked == "o":
            row = [
                InlineKeyboardButton("❌ Choose X", callback_data=f"ttt:sign:x:{game_id}"),
                InlineKeyboardButton("✅ 0️⃣ O Selected", callback_data=f"ttt:noop:{game_id}"),
            ]
        else:
            row = [
                InlineKeyboardButton("❌ Choose X", callback_data=f"ttt:sign:x:{game_id}"),
                InlineKeyboardButton("0️⃣ Choose O", callback_data=f"ttt:sign:o:{game_id}"),
            ]

        return InlineKeyboardMarkup(
            [
                row,
                [InlineKeyboardButton("🎲 Random", callback_data=f"ttt:sign:r:{game_id}")],
                [InlineKeyboardButton("❌ Cancel", callback_data=f"ttt:cancel:{game_id}")],
            ]
        )

    def active_keyboard(self, board: List[str], game_id: str, highlight: Optional[List[int]] = None) -> InlineKeyboardMarkup:
        highlight = highlight or []
        buttons: List[List[InlineKeyboardButton]] = []

        for r in range(3):
            row_btns: List[InlineKeyboardButton] = []
            for c in range(3):
                idx = r * 3 + c
                cell = board[idx]
                if not cell:
                    label = "⬜️"
                    cb = f"ttt:cell:{idx}:{game_id}"
                else:
                    sym = "❌" if cell == "X" else "0️⃣"
                    if idx in highlight:
                        sym = f"✨{sym}"
                    label = sym
                    cb = f"ttt:filled:{idx}:{game_id}"
                row_btns.append(InlineKeyboardButton(label, callback_data=cb))
            buttons.append(row_btns)

        buttons.append([InlineKeyboardButton("🏳️ Flee", callback_data=f"ttt:flee:{game_id}")])
        return InlineKeyboardMarkup(buttons)

    # --------------------- Logging ---------------------

    async def log_event(
        self,
        group_id: int,
        game_id: Optional[str],
        type_: str,
        payload: Dict[str, Any],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        body = {"groupId": group_id, "gameId": game_id, "type": type_, "payload": payload, "ts": utcnow()}
        self.mongo.log_event(body)

        group = self.mongo.groups.find_one({"_id": group_id})
        log_chat_id = group.get("logChatId") if group else None
        if log_chat_id:
            try:
                await context.bot.send_message(log_chat_id, f"[{type_}] {payload}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send log message: %s", exc)

    # --------------------- Background expiry ---------------------

    async def _expiry_loop(self, app: Application) -> None:
        # Cancels expired lobbies + optional turn timeouts.
        # Requires Mongo methods:
        # - list_expired_lobbies(now) -> list[game_docs]
        # - list_expired_turns(now) -> list[game_docs]   (optional)
        # - cancel_game(game_id, reason)
        # If you don't have them yet, implement with simple queries on games collection.
        while True:
            try:
                now = utcnow()

                # Expire lobbies
                for game in self.mongo.list_expired_lobbies(now):
                    gid = game["groupId"]
                    game_id = game["_id"]
                    # Clear lock + mark cancelled
                    self.mongo.record_result(game_id, {"type": "LOBBY_TIMEOUT", "winnerUserId": None, "line": None})
                    self.mongo.clear_active_game_if_match(gid, game_id)
                    self.mongo.update_game(game_id, {"$set": {"status": "CANCELLED", "updatedAt": now}})
                    try:
                        await app.bot.send_message(
                            gid,
                            build_card("Lobby expired", ["No one joined in time."], "Use /startgame to create a new lobby."),
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass
                    await self.log_event(gid, game_id, "lobby_timeout", {}, app.bot_data.get("ctx") or None)  # safe no-op

                # Optional: Turn timeouts (if you store lastMoveAt / updatedAt)
                for game in self.mongo.list_expired_turns(now):
                    gid = game["groupId"]
                    game_id = game["_id"]
                    if game.get("status") != "ACTIVE":
                        continue
                    turn = game.get("turn")
                    loser = game["playerX"] if turn == "X" else game["playerO"]
                    winner = game["playerO"] if turn == "X" else game["playerX"]
                    self.mongo.record_result(game_id, {"type": "TURN_TIMEOUT", "winnerUserId": winner, "line": None})
                    self.mongo.clear_active_game_if_match(gid, game_id)
                    self.mongo.update_game(game_id, {"$set": {"status": "ENDED", "updatedAt": now}})
                    self.mongo.bump_stats(winner, loser, draw=False)
                    try:
                        await app.bot.send_message(
                            gid,
                            build_card("Game over", [f"Winner (timeout): {winner}"], "Use /startgame for a new lobby."),
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

                await asyncio.sleep(10)

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("Expiry loop error: %s", exc)
                await asyncio.sleep(10)

    async def post_init(self, app: Application) -> None:
        # Store a fake context handle for log_event calls from expiry loop
        app.bot_data["ctx"] = None
        self._expiry_task = asyncio.create_task(self._expiry_loop(app))

    async def post_shutdown(self, app: Application) -> None:
        if self._expiry_task:
            self._expiry_task.cancel()
            with contextlib.suppress(Exception):
                await self._expiry_task

    # --------------------- Commands ---------------------

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat and update.effective_chat.type == ChatType.PRIVATE:
            await update.effective_message.reply_html(
                build_card(
                    "TicTacToe Bot",
                    [
                        "Add me to a group and run /startgame",
                        "One active game per group • Stats • Logs • Rematch",
                    ],
                    "Group commands: /startgame /join /stats /groupstats",
                )
            )
        else:
            await update.effective_message.reply_text("Use /startgame to begin a lobby.")

    async def startgame(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if not msg or not chat or not user:
            return
        if chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            await msg.reply_text("Please use me in a group to play.")
            return

        self.mongo.upsert_user({"_id": user.id, "name": user.full_name, "username": user.username})
        group = self.mongo.get_or_create_group({"id": chat.id, "title": chat.title})

        if group.get("activeGameId"):
            await msg.reply_html(
                build_card(
                    "Game already active",
                    ["Only one lobby or game is allowed per group."],
                    "Wait for it to finish, or admins can /forceend.",
                )
            )
            return

        # Create game id and attempt to lock group first (prevents orphans)
        game_id = str(uuid.uuid4())
        if not self.mongo.set_active_game_if_free(chat.id, game_id):
            await msg.reply_text("Another game was just created. Try again.")
            return

        now = utcnow()
        join_until = now + dt.timedelta(seconds=self.settings.join_timeout_sec)

        game_doc = {
            "_id": game_id,
            "groupId": chat.id,
            "status": "LOBBY",
            "createdAt": now,
            "updatedAt": now,
            "createdBy": user.id,
            "hostId": user.id,
            "joinMode": "open",                 # open | invite
            "invitedUser": None,                # user id (optional)
            "invitedUsername": None,            # username (optional fallback)
            "joinExpiresAt": join_until,
            "playerX": None,
            "playerO": None,
            "players": [user.id],
            "board": ["" for _ in range(9)],
            "turn": None,
            "moves": [],
            "messageRefs": {},
            "result": None,
        }

        try:
            self.mongo.create_game(game_doc)
        except Exception:
            # rollback lock if insert fails
            self.mongo.clear_active_game_if_match(chat.id, game_id)
            raise

        text = build_card(
            "🎮 TicTacToe Lobby",
            [
                f"Host: {user.mention_html()}",
                "Challenger: <i>Waiting…</i>",
                f"Expires in: <b>{self.settings.join_timeout_sec}s</b>",
            ],
            "Use /join or press ✅ Join Game",
        )
        sent = await msg.reply_html(text, reply_markup=self.lobby_keyboard(game_id))
        self.mongo.update_game(game_id, {"$set": {"messageRefs.lobbyMsgId": sent.message_id}})
        await self.log_event(chat.id, game_id, "lobby_created", {"host": user.id}, context)

    async def _join_flow(
        self,
        chat,
        user,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        args: Optional[List[str]] = None,
        message=None,
        reply_to_message=None,
    ) -> None:
        """Shared join flow for /join command and Join Game button."""

        if not chat or not user:
            return

        async def _send_text(text: str) -> None:
            if message:
                await message.reply_text(text)
            else:
                await context.bot.send_message(chat.id, text)

        async def _send_html(text: str, *, markup=None) -> None:
            if message:
                await message.reply_html(text, reply_markup=markup)
            else:
                await context.bot.send_message(chat.id, text, parse_mode=ParseMode.HTML, reply_markup=markup)

        if chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            await _send_text("Join works inside a group lobby.")
            return

        game = self.current_game(chat.id)
        if not game or game.get("status") not in {"LOBBY", "SIGN_PICK"}:
            await _send_text("No active lobby. Use /startgame first.")
            return

        self.mongo.upsert_user({"_id": user.id, "name": user.full_name, "username": user.username})

        # Invite flow:
        # Preferred: reply to a user's message -> we get user_id
        # Optional: /join @username -> store invitedUsername, accept checks query.from_user.username
        target_user_id: Optional[int] = None
        target_username: Optional[str] = None

        if reply_to_message and reply_to_message.from_user:
            target_user_id = reply_to_message.from_user.id

        if args and args[0].startswith("@"):
            target_username = args[0].lstrip("@").strip() or None

        if (target_user_id or target_username) and (game.get("status") == "LOBBY"):
            join_until = utcnow() + dt.timedelta(seconds=self.settings.join_timeout_sec)

            if target_user_id and target_user_id == user.id:
                await _send_text("You can't invite yourself.")
                return

            self.mongo.update_game(
                game["_id"],
                {
                    "$set": {
                        "joinMode": "invite",
                        "invitedUser": target_user_id,
                        "invitedUsername": target_username,
                        "joinExpiresAt": join_until,
                        "updatedAt": utcnow(),
                    }
                },
            )

            invite_line = (
                f"Invited: {mention_html(target_user_id, 'Player')}" if target_user_id
                else f"Invited username: <code>@{target_username}</code>"
            )

            text = build_card(
                "📨 Invitation sent",
                [
                    f"Host: {user.mention_html()}",
                    invite_line,
                    f"Expires in: <b>{self.settings.join_timeout_sec}s</b>",
                ],
                "Only the invited user can accept.",
            )
            await _send_html(text, markup=self.invite_keyboard(game["_id"]))
            await self.log_event(chat.id, game["_id"], "invite_sent", {"to": target_user_id, "toUsername": target_username, "by": user.id}, context)
            return

        # Normal join (open lobby)
        if user.id in game.get("players", []):
            await _send_text("You are already in the lobby.")
            return

        players = game.get("players", [])
        if len(players) >= 2:
            await _send_text("Lobby already has two players.")
            return

        players.append(user.id)
        new_status = "SIGN_PICK" if len(players) == 2 else "LOBBY"
        updated = self.mongo.update_game(game["_id"], {"$set": {"players": players, "status": new_status, "updatedAt": utcnow()}})

        await self.log_event(chat.id, game["_id"], "player_join", {"player": user.id}, context)

        # Edit lobby message instead of sending new noise
        await self.render_lobby_or_sign(updated, context)

    async def join(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if not msg or not chat or not user:
            return

        await self._join_flow(
            chat,
            user,
            context,
            args=context.args,
            message=msg,
            reply_to_message=msg.reply_to_message,
        )

    async def render_lobby_or_sign(self, game: Dict, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = game["groupId"]
        lobby_msg_id = game.get("messageRefs", {}).get("lobbyMsgId")

        players = game.get("players", [])
        host_id = players[0] if players else game.get("hostId")

        # Show lobby or sign pick depending on state
        if game.get("status") == "SIGN_PICK" and len(players) >= 2:
            await self.prompt_sign_choice(chat_id, lobby_msg_id, game, context)
            return

        body = [
            f"Host: <code>{host_id}</code>",
            f"Challenger: <i>{'Waiting…' if len(players) < 2 else players[1]}</i>",
        ]
        if game.get("joinExpiresAt"):
            remain = int((game["joinExpiresAt"] - utcnow()).total_seconds())
            remain = max(remain, 0)
            body.append(f"Expires in: <b>{remain}s</b>")

        text = build_card("🎮 TicTacToe Lobby", body, "Use /join or press ✅ Join Game")
        markup = self.lobby_keyboard(game["_id"])

        if lobby_msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=lobby_msg_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
                return
            except Exception:
                pass

        sent = await context.bot.send_message(chat_id, text, parse_mode=ParseMode.HTML, reply_markup=markup)
        self.mongo.update_game(game["_id"], {"$set": {"messageRefs.lobbyMsgId": sent.message_id}})

    async def prompt_sign_choice(self, chat_id: int, lobby_msg_id: Optional[int], game: Dict, context: ContextTypes.DEFAULT_TYPE) -> None:
        players = game.get("players", [])
        if len(players) < 2:
            return

        host_id = game.get("hostId") or players[0]
        text = build_card(
            "🎭 Choose your sign",
            [
                f"Players: <code>{players[0]}</code> vs <code>{players[1]}</code>",
                f"Host (<code>{host_id}</code>) picks ❌ or 0️⃣; the challenger gets the other.",
            ],
            "Tap ❌ or 0️⃣ (or 🎲 Random)",
        )
        markup = self.sign_keyboard(game["_id"])

        if lobby_msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=lobby_msg_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
                return
            except Exception:
                logger.info("Could not edit lobby message; sending new sign prompt")

        sent = await context.bot.send_message(chat_id, text, parse_mode=ParseMode.HTML, reply_markup=markup)
        self.mongo.update_game(game["_id"], {"$set": {"messageRefs.lobbyMsgId": sent.message_id}})

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user:
            return
        doc = self.mongo.user_stats(update.effective_user.id) or {}
        stats = doc.get("stats") or {"wins": 0, "losses": 0, "draws": 0, "gamesPlayed": 0}
        body = [
            f"Wins: {stats.get('wins', 0)}",
            f"Losses: {stats.get('losses', 0)}",
            f"Draws: {stats.get('draws', 0)}",
            f"Games: {stats.get('gamesPlayed', 0)}",
        ]
        await update.effective_message.reply_html(build_card("Your stats", body))

    async def groupstats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat:
            return
        stats = self.mongo.group_stats(update.effective_chat.id)
        body = [
            f"Total games: {stats.get('total', 0)}",
            f"Wins: {stats.get('wins', 0)}",
            f"Draws: {stats.get('draws', 0)}",
            f"Losses: {stats.get('losses', 0)}",
        ]
        await update.effective_message.reply_html(build_card("Group stats", body))

    async def broadcast(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not is_owner(update.effective_user.id, self.settings):
            await update.effective_message.reply_text("Owner only")
            return
        if not context.args:
            await update.effective_message.reply_text("Usage: /broadcast <message>")
            return
        msg = " ".join(context.args)

        successes, fails = 0, 0
        for gid in self.mongo.all_group_ids():
            try:
                await context.bot.send_message(gid, msg)
                successes += 1
                await asyncio.sleep(0.25)
            except RetryAfter as e:
                await asyncio.sleep(float(e.retry_after) + 0.2)
            except (TimedOut, Exception):  # noqa: BLE001
                fails += 1

        await update.effective_message.reply_text(f"Broadcast done ✅ {successes} / ❌ {fails}")

    async def setloggroup(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.effective_user:
            return
        member = await update.effective_chat.get_member(update.effective_user.id)
        if member.status not in {"administrator", "creator"} and not is_owner(update.effective_user.id, self.settings):
            await update.effective_message.reply_text("Admins only.")
            return
        if not context.args:
            await update.effective_message.reply_text("Usage: /setloggroup <chat_id|null>")
            return
        arg = context.args[0]
        log_chat_id = None if arg.lower() == "null" else int(arg)
        self.mongo.set_log_group(update.effective_chat.id, log_chat_id)
        await update.effective_message.reply_text("Log destination updated ✅")

    async def forceend(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.effective_user:
            return
        member = await update.effective_chat.get_member(update.effective_user.id)
        if member.status not in {"administrator", "creator"} and not is_owner(update.effective_user.id, self.settings):
            await update.effective_message.reply_text("Admins only.")
            return

        game = self.current_game(update.effective_chat.id)
        if not game:
            await update.effective_message.reply_text("No active game.")
            return

        self.mongo.record_result(game["_id"], {"type": "ADMIN_END", "winnerUserId": None, "line": None})
        self.mongo.update_game(game["_id"], {"$set": {"status": "ENDED", "updatedAt": utcnow()}})
        self.mongo.clear_active_game_if_match(update.effective_chat.id, game["_id"])
        await update.effective_message.reply_text("Game forcibly ended ✅")
        await self.log_event(update.effective_chat.id, game["_id"], "forceend", {"by": update.effective_user.id}, context)

    # --------------------- Callbacks ---------------------

    async def noop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if q:
            await q.answer("Not available.", show_alert=False)

    async def filled_cell(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if q:
            await q.answer("Cell already taken.", show_alert=False)

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not q.data:
            return
        if self._throttle(q.from_user.id):
            await q.answer("Slow down")
            return

        parts = q.data.split(":")
        game_id = parts[2]
        game = self.mongo.fetch_game(game_id)
        if not game:
            await q.answer("Lobby not found", show_alert=True)
            return

        chat_id = game["groupId"]

        # Permission: host/admin/owner can cancel
        allowed = False
        try:
            member = await context.bot.get_chat_member(chat_id, q.from_user.id)
            allowed = member.status in {"administrator", "creator"} or q.from_user.id == game.get("hostId")
        except Exception:
            allowed = q.from_user.id == game.get("hostId")

        if not allowed and not is_owner(q.from_user.id, self.settings):
            await q.answer("Only host/admin can cancel.", show_alert=True)
            return

        self.mongo.record_result(game_id, {"type": "CANCELLED", "winnerUserId": None, "line": None})
        self.mongo.update_game(game_id, {"$set": {"status": "CANCELLED", "updatedAt": utcnow()}})
        self.mongo.clear_active_game_if_match(chat_id, game_id)
        await q.answer("Cancelled ✅")
        await context.bot.send_message(chat_id, build_card("Lobby cancelled", ["Lobby closed."], "Use /startgame to play again."), parse_mode=ParseMode.HTML)
        await self.log_event(chat_id, game_id, "cancelled", {"by": q.from_user.id}, context)

    async def accept_invite(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not q.data:
            return
        if self._throttle(q.from_user.id):
            await q.answer("Slow")
            return

        game_id = q.data.split(":")[2]
        game = self.mongo.fetch_game(game_id)
        if not game or game.get("status") != "LOBBY":
            await q.answer("Lobby expired", show_alert=True)
            return

        # Validate invite by user id OR username
        invited_id = game.get("invitedUser")
        invited_username = game.get("invitedUsername")
        if invited_id and invited_id != q.from_user.id:
            await q.answer("Not your invite", show_alert=True)
            return
        if (not invited_id) and invited_username:
            if not q.from_user.username or q.from_user.username.lower() != invited_username.lower():
                await q.answer("Not your invite", show_alert=True)
                return

        if q.from_user.id in game.get("players", []):
            await q.answer("You are already in this lobby.", show_alert=True)
            return

        players = game.get("players", [])
        if len(players) >= 2:
            await q.answer("Lobby full", show_alert=True)
            return

        players.append(q.from_user.id)
        updated = self.mongo.update_game(game_id, {"$set": {"players": players, "status": "SIGN_PICK", "updatedAt": utcnow()}})

        await q.answer("Joined! Pick a sign ✅")
        await self.render_lobby_or_sign(updated, context)
        await self.log_event(game["groupId"], game_id, "invite_accepted", {"by": q.from_user.id}, context)

    async def decline_invite(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not q.data:
            return
        if self._throttle(q.from_user.id):
            await q.answer("Slow")
            return

        game_id = q.data.split(":")[2]
        game = self.mongo.fetch_game(game_id)
        if not game:
            await q.answer("Lobby gone", show_alert=True)
            return

        # Only invited user can decline
        invited_id = game.get("invitedUser")
        invited_username = game.get("invitedUsername")
        allowed = (invited_id == q.from_user.id) or (invited_username and q.from_user.username and q.from_user.username.lower() == invited_username.lower())
        if not allowed:
            await q.answer("Not your invite", show_alert=True)
            return

        self.mongo.record_result(game_id, {"type": "INVITE_DECLINED", "winnerUserId": None, "line": None})
        self.mongo.update_game(game_id, {"$set": {"status": "CANCELLED", "updatedAt": utcnow()}})
        self.mongo.clear_active_game_if_match(game["groupId"], game_id)

        await q.answer("Declined")
        await context.bot.send_message(game["groupId"], build_card("Invite declined", ["Lobby closed."], "Use /startgame to play again."), parse_mode=ParseMode.HTML)
        await self.log_event(game["groupId"], game_id, "invite_declined", {"by": q.from_user.id}, context)

    async def handle_sign(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not q.data:
            return
        if self._throttle(q.from_user.id):
            await q.answer("Slow down")
            return

        _, _, choice, game_id = q.data.split(":")
        game = self.mongo.fetch_game(game_id)
        if not game or game.get("status") != "SIGN_PICK":
            await q.answer("Game not available.", show_alert=True)
            return

        players = game.get("players", [])
        if q.from_user.id not in players:
            await q.answer("Only players can pick.", show_alert=True)
            return

        host_id = game.get("hostId") or (players[0] if players else None)
        if host_id and q.from_user.id != host_id:
            await q.answer("Only the host can choose the sign.", show_alert=True)
            return

        if game.get("playerX") or game.get("playerO"):
            await q.answer("Sign already chosen.", show_alert=True)
            return

        chooser = q.from_user.id
        other = players[0] if players[1] == chooser else players[1]

        if choice == "r":
            choice = random.choice(["x", "o"])

        locked = "x" if choice == "x" else "o"
        # Immediate UI lock feel
        lobby_msg_id = game.get("messageRefs", {}).get("lobbyMsgId")
        if lobby_msg_id:
            try:
                text = build_card(
                    "🎭 Choose your sign",
                    [
                        f"Players: <code>{players[0]}</code> vs <code>{players[1]}</code>",
                        "Locking selection…",
                    ],
                )
                await context.bot.edit_message_text(
                    chat_id=game["groupId"],
                    message_id=lobby_msg_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=self.sign_keyboard(game_id, locked=locked),
                )
            except Exception:
                pass

        if choice == "x":
            player_x, player_o = chooser, other
        else:
            player_o, player_x = chooser, other

        updated = self.mongo.update_game(
            game_id,
            {"$set": {"playerX": player_x, "playerO": player_o, "turn": "X", "status": "ACTIVE", "updatedAt": utcnow()}},
        )

        await q.answer("Signs locked ✅ Game started!")
        await self.start_board(updated, context)
        await self.log_event(game["groupId"], game_id, "sign_selected", {"playerX": player_x, "playerO": player_o, "by": chooser}, context)

    async def start_board(self, game: Dict, context: ContextTypes.DEFAULT_TYPE) -> None:
        board = game.get("board") or ["" for _ in range(9)]
        caption = build_card(
            "🎮 TicTacToe",
            [
                f"❌ <code>{game['playerX']}</code>  vs  0️⃣ <code>{game['playerO']}</code>",
                f"Turn: ❌ (<code>{game['playerX']}</code>)",
            ],
            "Tap a cell to move",
        )
        markup = self.active_keyboard(board, game["_id"])
        msg = await context.bot.send_message(
            game["groupId"],
            caption,
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )
        self.mongo.update_game(game["_id"], {"$set": {"messageRefs.gameMsgId": msg.message_id}})

    async def handle_cell(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not q.data:
            return
        if self._throttle(q.from_user.id):
            await q.answer("Hold on")
            return

        parts = q.data.split(":")
        if len(parts) < 4:
            await q.answer("Malformed move", show_alert=True)
            return

        idx = int(parts[2])
        game_id = parts[3]

        game = self.mongo.fetch_game(game_id)
        if not game or game.get("status") != "ACTIVE":
            await q.answer("Game finished.", show_alert=True)
            return

        group = self.mongo.groups.find_one({"_id": game["groupId"]})
        if group and group.get("activeGameId") != game_id:
            await q.answer("Stale game.", show_alert=True)
            return

        # Determine player symbol
        if q.from_user.id == game.get("playerX"):
            symbol = "X"
        elif q.from_user.id == game.get("playerO"):
            symbol = "O"
        else:
            await q.answer("You are not in this game.", show_alert=True)
            return

        if game.get("turn") != symbol:
            await q.answer("Not your turn.", show_alert=False)
            return

        board = game.get("board") or ["" for _ in range(9)]
        if board[idx]:
            await q.answer("Cell already taken.", show_alert=False)
            return

        board[idx] = symbol
        next_turn = "O" if symbol == "X" else "X"
        move = {"by": q.from_user.id, "symbol": symbol, "idx": idx, "ts": utcnow()}

        self.mongo.update_game(
            game_id,
            {
                "$set": {"board": board, "turn": next_turn, "updatedAt": utcnow()},
                "$push": {"moves": move},
            },
        )

        winner, line = check_winner(board)
        chat_id = game["groupId"]
        msg_id = game.get("messageRefs", {}).get("gameMsgId")

        if winner == "DRAW":
            caption = build_card(
                "🎮 TicTacToe",
                [f"❌ <code>{game['playerX']}</code>  vs  0️⃣ <code>{game['playerO']}</code>", "Result: <b>Draw</b>"],
                "Use 🔄 Rematch or /startgame",
            )
            markup = self.active_keyboard(board, game_id, highlight=[])
            # End game
            self.mongo.record_result(game_id, {"type": "DRAW", "winnerUserId": None, "line": None})
            self.mongo.update_game(game_id, {"$set": {"status": "ENDED", "updatedAt": utcnow()}})
            self.mongo.clear_active_game_if_match(chat_id, game_id)

            # Ensure both players stats update (no None crash)
            self.mongo.bump_stats(game["playerX"], game["playerO"], draw=True)

            await self._edit_or_send_board(chat_id, msg_id, caption, markup=None, context=context)
            await self.log_event(chat_id, game_id, "game_ended", {"draw": True}, context)
            await self._send_end_buttons(chat_id, game_id, context)
            await q.answer()
            return

        if winner in {"X", "O"}:
            winner_id = game["playerX"] if winner == "X" else game["playerO"]
            loser_id = game["playerO"] if winner == "X" else game["playerX"]

            caption = build_card(
                "🏁 Game Over",
                [
                    f"Winner: <code>{winner_id}</code>",
                    f"Line: <code>{line}</code>",
                ],
                "Use 🔄 Rematch or /startgame",
            )
            end_markup = self.active_keyboard(board, game_id, highlight=line or [])

            self.mongo.record_result(game_id, {"type": "WIN", "winnerUserId": winner_id, "line": line})
            self.mongo.update_game(game_id, {"$set": {"status": "ENDED", "updatedAt": utcnow()}})
            self.mongo.clear_active_game_if_match(chat_id, game_id)
            self.mongo.bump_stats(winner_id, loser_id, draw=False)

            await self._edit_or_send_board(chat_id, msg_id, caption, markup=None, context=context)  # no clicks after end
            # Optionally send a final highlighted board view
            try:
                await context.bot.send_message(chat_id, "✨ Final Board:", reply_markup=end_markup)
            except Exception:
                pass

            await self.log_event(chat_id, game_id, "game_ended", {"winner": winner_id, "draw": False}, context)
            await self._send_end_buttons(chat_id, game_id, context)
            await q.answer()
            return

        # Normal continue
        turn_user = game["playerX"] if next_turn == "X" else game["playerO"]
        caption = build_card(
            "🎮 TicTacToe",
            [
                f"❌ <code>{game['playerX']}</code>  vs  0️⃣ <code>{game['playerO']}</code>",
                f"Turn: {'❌' if next_turn == 'X' else '0️⃣'} (<code>{turn_user}</code>)",
            ],
            "Tap a cell to move",
        )
        markup = self.active_keyboard(board, game_id)
        await self._edit_or_send_board(chat_id, msg_id, caption, markup=markup, context=context)
        await q.answer()

    async def _edit_or_send_board(
        self,
        chat_id: int,
        msg_id: Optional[int],
        caption: str,
        markup: Optional[InlineKeyboardMarkup],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
                return
            except Exception:
                pass
        await context.bot.send_message(chat_id, caption, parse_mode=ParseMode.HTML, reply_markup=markup)

    async def _send_end_buttons(self, chat_id: int, game_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Rematch", callback_data=f"ttt:rematch:{game_id}")],
                [InlineKeyboardButton("🏁 New Game", callback_data=f"ttt:new:{game_id}")],
            ]
        )
        await context.bot.send_message(chat_id, "What next?", reply_markup=kb)

    async def flee(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not q.data:
            return
        if self._throttle(q.from_user.id):
            await q.answer("Slow")
            return

        game_id = q.data.split(":")[2]
        game = self.mongo.fetch_game(game_id)
        if not game:
            await q.answer("No active game", show_alert=True)
            return

        chat_id = game["groupId"]
        status = game.get("status")

        if status == "LOBBY":
            # Treat as cancel lobby (host/admin only)
            await self.cancel(update, context)
            return

        if status != "ACTIVE":
            await q.answer("Game already ended.", show_alert=True)
            return

        # Only players can flee
        if q.from_user.id not in {game.get("playerX"), game.get("playerO")}:
            await q.answer("Players only.", show_alert=True)
            return

        opponent = game["playerO"] if q.from_user.id == game.get("playerX") else game["playerX"]

        self.mongo.record_result(game_id, {"type": "FORFEIT", "winnerUserId": opponent, "line": None})
        self.mongo.update_game(game_id, {"$set": {"status": "ENDED", "updatedAt": utcnow()}})
        self.mongo.clear_active_game_if_match(chat_id, game_id)
        self.mongo.bump_stats(opponent, q.from_user.id, draw=False)

        await q.answer("You fled.")
        await context.bot.send_message(
            chat_id,
            build_card("🏁 Game Over", [f"Winner (forfeit): <code>{opponent}</code>"], "Use /startgame for a new lobby."),
            parse_mode=ParseMode.HTML,
        )
        await self.log_event(chat_id, game_id, "forfeit", {"by": q.from_user.id, "winner": opponent}, context)

    async def rematch(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not q.data:
            return
        if self._throttle(q.from_user.id):
            await q.answer("Slow")
            return

        game_id = q.data.split(":")[2]
        prev = self.mongo.fetch_game(game_id)
        if not prev:
            await q.answer("Game missing", show_alert=True)
            return

        players = prev.get("players", [])
        if q.from_user.id not in players:
            await q.answer("Players only", show_alert=True)
            return

        group_id = prev["groupId"]
        group = self.mongo.groups.find_one({"_id": group_id}) or {}
        if group.get("activeGameId"):
            await q.answer("Another game is active.", show_alert=True)
            return

        new_id = str(uuid.uuid4())
        if not self.mongo.set_active_game_if_free(group_id, new_id):
            await q.answer("Could not lock rematch", show_alert=True)
            return

        now = utcnow()
        new_game = {
            "_id": new_id,
            "groupId": group_id,
            "status": "SIGN_PICK",
            "createdAt": now,
            "updatedAt": now,
            "createdBy": q.from_user.id,
            "hostId": players[0],
            "joinMode": "open",
            "invitedUser": None,
            "invitedUsername": None,
            "joinExpiresAt": None,
            "playerX": None,
            "playerO": None,
            "players": players[:2],
            "board": ["" for _ in range(9)],
            "turn": None,
            "moves": [],
            "messageRefs": {},
            "result": None,
        }
        self.mongo.create_game(new_game)

        await q.answer("Rematch! Pick signs ✅")
        await self.render_lobby_or_sign(new_game, context)

    async def new_game_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if q:
            await q.answer()
            await context.bot.send_message(q.message.chat_id, "Use /startgame to begin a new lobby.")

    async def fallback_join_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not q.message:
            return
        if self._throttle(q.from_user.id):
            await q.answer()
            return
        await q.answer()

        await self._join_flow(
            q.message.chat,
            q.from_user,
            context,
            args=None,
            message=q.message,
            reply_to_message=None,
        )

    # --------------------- Build app ---------------------

    def build_app(self) -> Application:
        app = (
            Application.builder()
            .token(self.settings.bot_token)
            .rate_limiter(AIORateLimiter())
            .post_init(self.post_init)
            .build()
        )

        # Commands
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("startgame", self.startgame))
        app.add_handler(CommandHandler("join", self.join))
        app.add_handler(CommandHandler("stats", self.stats))
        app.add_handler(CommandHandler("groupstats", self.groupstats))
        app.add_handler(CommandHandler("broadcast", self.broadcast))
        app.add_handler(CommandHandler("forceend", self.forceend))
        app.add_handler(CommandHandler("setloggroup", self.setloggroup))

        # Callbacks
        app.add_handler(CallbackQueryHandler(self.handle_sign, pattern=r"^ttt:sign:"))
        app.add_handler(CallbackQueryHandler(self.handle_cell, pattern=r"^ttt:cell:"))
        app.add_handler(CallbackQueryHandler(self.filled_cell, pattern=r"^ttt:filled:"))
        app.add_handler(CallbackQueryHandler(self.accept_invite, pattern=r"^ttt:accept:"))
        app.add_handler(CallbackQueryHandler(self.decline_invite, pattern=r"^ttt:decline:"))
        app.add_handler(CallbackQueryHandler(self.flee, pattern=r"^ttt:flee:"))
        app.add_handler(CallbackQueryHandler(self.cancel, pattern=r"^ttt:cancel:"))
        app.add_handler(CallbackQueryHandler(self.rematch, pattern=r"^ttt:rematch:"))
        app.add_handler(CallbackQueryHandler(self.new_game_button, pattern=r"^ttt:new:"))
        app.add_handler(CallbackQueryHandler(self.fallback_join_button, pattern=r"^ttt:join:"))
        app.add_handler(CallbackQueryHandler(self.noop, pattern=r"^ttt:noop:"))

        return app


def main() -> None:
    settings = load_settings()
    bot = TicToeBot(settings)
    app = bot.build_app()
    logger.info("Starting polling bot")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
