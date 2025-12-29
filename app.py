"""Telegram TicTacToe bot using python-telegram-bot v20."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from cachetools import TTLCache
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import AIORateLimiter, Application, CallbackQueryHandler, CommandHandler, ContextTypes

from config import is_owner, load_settings
from mongodb import Mongo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Helpers ---------------------------------------------------------------

def build_card(title: str, body: List[str], footer: Optional[str] = None) -> str:
    lines = [f"<b>{title}</b>", "──────────────────"]
    lines.extend(body)
    if footer:
        lines.append("──────────────────")
        lines.append(footer)
    return "\n".join(lines)


def board_markup(board: List[str], game_id: str, highlight: Optional[List[int]] = None) -> Tuple[str, InlineKeyboardMarkup]:
    highlight = highlight or []
    buttons: List[List[InlineKeyboardButton]] = []
    rows = []
    for r in range(3):
        row_btns = []
        row_symbols = []
        for c in range(3):
            idx = r * 3 + c
            cell = board[idx]
            symbol = "⬜️" if not cell else ("❌" if cell == "X" else "0️⃣")
            if idx in highlight and cell:
                symbol = f"✨{symbol}"
            row_symbols.append(symbol)
            row_btns.append(InlineKeyboardButton(symbol, callback_data=f"ttt:cell:{idx}:{game_id}"))
        rows.append("".join(row_symbols))
        buttons.append(row_btns)
    text = "\n".join(rows)
    return text, InlineKeyboardMarkup(buttons)


def check_winner(board: List[str]) -> Tuple[Optional[str], Optional[List[int]]]:
    lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    for a, b, c in lines:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a], [a, b, c]
    if all(board):
        return "DRAW", None
    return None, None


# --- Bot class ------------------------------------------------------------


class TicToeBot:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.mongo = Mongo(settings)
        self.throttle_cache: TTLCache[int, float] = TTLCache(maxsize=1000, ttl=settings.callback_throttle_sec)

    # Utility
    def _throttle(self, user_id: int) -> bool:
        if user_id in self.throttle_cache:
            return True
        self.throttle_cache[user_id] = dt.datetime.utcnow().timestamp()
        return False

    def current_game(self, group_id: int) -> Optional[Dict]:
        group = self.mongo.groups.find_one({"_id": group_id})
        if not group:
            return None
        game_id = group.get("activeGameId")
        if not game_id:
            return None
        return self.mongo.fetch_game(game_id)

    # UI helpers
    def lobby_keyboard(self, game_id: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Join Game", callback_data=f"ttt:join:{game_id}")],
                [InlineKeyboardButton("🏳️ Flee", callback_data=f"ttt:flee:{game_id}")],
            ]
        )

    def sign_keyboard(self, game_id: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("❌ Choose X", callback_data=f"ttt:sign:x:{game_id}"),
                    InlineKeyboardButton("0️⃣ Choose O", callback_data=f"ttt:sign:o:{game_id}"),
                ],
                [InlineKeyboardButton("🎲 Random", callback_data=f"ttt:sign:r:{game_id}")],
            ]
        )

    async def log_event(self, group_id: int, game_id: Optional[str], type_: str, payload: Dict[str, Any], context: ContextTypes.DEFAULT_TYPE) -> None:
        body = {"groupId": group_id, "gameId": game_id, "type": type_, "payload": payload}
        self.mongo.log_event(body)
        group = self.mongo.groups.find_one({"_id": group_id})
        log_chat_id = group.get("logChatId") if group else None
        if log_chat_id:
            try:
                await context.bot.send_message(log_chat_id, f"[{type_}] {payload}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send log message: %s", exc)

    # Command handlers
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat and update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_html(
                build_card(
                    "TicToe Bot",
                    [
                        "Add me to a group and run /startgame",
                        "One active game per group with stats and logs.",
                    ],
                )
            )
        else:
            await update.message.reply_text("Hello! Use /startgame to begin a lobby.")

    async def startgame(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message or not update.effective_chat:
            return
        if update.effective_chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            await message.reply_text("Please use me in a group to play.")
            return
        self.mongo.upsert_user(
            {
                "_id": update.effective_user.id,
                "name": update.effective_user.full_name,
                "username": update.effective_user.username,
            }
        )
        group = self.mongo.get_or_create_group({"id": update.effective_chat.id, "title": update.effective_chat.title})
        if group.get("activeGameId"):
            await message.reply_html(
                build_card(
                    "Game already active",
                    ["Only one lobby or game is allowed per group."],
                    "Wait for it to finish or /forceend if you are an admin.",
                )
            )
            return

        game_id = str(uuid.uuid4())
        now = dt.datetime.utcnow()
        game_doc = {
            "_id": game_id,
            "groupId": update.effective_chat.id,
            "status": "LOBBY",
            "createdAt": now,
            "updatedAt": now,
            "createdBy": update.effective_user.id,
            "joinMode": "open",
            "invitedUser": None,
            "joinExpiresAt": None,
            "playerX": None,
            "playerO": None,
            "players": [update.effective_user.id],
            "board": ["" for _ in range(9)],
            "turn": None,
            "moves": [],
            "messageRefs": {},
            "result": None,
        }
        self.mongo.create_game(game_doc)
        if not self.mongo.set_active_game_if_free(update.effective_chat.id, game_id):
            await message.reply_text("Another game was just created. Try again shortly.")
            return

        text = build_card(
            "TicToe Lobby",
            [
                f"Host: {update.effective_user.mention_html()}",
                "Waiting for challenger...",
            ],
            "Use /join or press the button.",
        )
        sent = await message.reply_html(text, reply_markup=self.lobby_keyboard(game_id))
        self.mongo.update_game(game_id, {"$set": {"messageRefs.lobbyMsgId": sent.message_id}})
        await self.log_event(update.effective_chat.id, game_id, "lobby_created", {"host": update.effective_user.id}, context)

    async def join(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message or not update.effective_chat:
            return
        if update.effective_chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            await message.reply_text("Join only works inside a group lobby.")
            return
        group_id = update.effective_chat.id
        game = self.current_game(group_id)
        if not game or game.get("status") not in {"LOBBY", "SIGN_PICK"}:
            await message.reply_text("No active lobby. Use /startgame first.")
            return

        from_user = update.effective_user
        self.mongo.upsert_user({"_id": from_user.id, "name": from_user.full_name, "username": from_user.username})

        target_user: Optional[int] = None
        if message.reply_to_message and message.reply_to_message.from_user:
            target_user = message.reply_to_message.from_user.id

        if target_user and target_user != from_user.id:
            join_until = dt.datetime.utcnow() + dt.timedelta(seconds=self.settings.join_timeout_sec)
            self.mongo.update_game(
                game["_id"],
                {"$set": {"joinMode": "invite", "invitedUser": target_user, "joinExpiresAt": join_until}},
            )
            text = build_card(
                "Invitation sent",
                [
                    f"Host: <b>{from_user.full_name}</b>",
                    f"Invited: <code>{target_user}</code>",
                    "Awaiting acceptance...",
                ],
            )
            buttons = InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("✅ Accept", callback_data=f"ttt:accept:{game['_id']}")],
                    [InlineKeyboardButton("❌ Decline", callback_data=f"ttt:decline:{game['_id']}")],
                ]
            )
            await message.reply_html(text, reply_markup=buttons)
            await self.log_event(group_id, game["_id"], "invite_sent", {"to": target_user, "by": from_user.id}, context)
            return

        if from_user.id in game.get("players", []):
            await message.reply_text("You are already in the lobby.")
            return
        players = game.get("players", [])
        if len(players) >= 2:
            await message.reply_text("Lobby already has two players.")
            return
        players.append(from_user.id)
        new_status = "SIGN_PICK" if len(players) == 2 else "LOBBY"
        updated = self.mongo.update_game(game["_id"], {"$set": {"players": players, "status": new_status}})
        await self.log_event(group_id, game["_id"], "player_join", {"player": from_user.id}, context)
        if new_status == "SIGN_PICK":
            await self.prompt_sign_choice(update, context, updated)
        else:
            await message.reply_text(f"{from_user.mention_html()} joined the lobby!", parse_mode=ParseMode.HTML)

    async def prompt_sign_choice(self, update: Update, context: ContextTypes.DEFAULT_TYPE, game: Dict) -> None:
        chat_id = game["groupId"]
        lobby_msg_id = game.get("messageRefs", {}).get("lobbyMsgId")
        players = game.get("players", [])
        if len(players) < 2:
            return
        p1, p2 = players[0], players[1]
        text = build_card(
            "Choose your sign",
            [f"{p1} vs {p2}", "First choice locks the mapping."],
            "Tap X or O",
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
        await context.bot.send_message(chat_id, text, parse_mode=ParseMode.HTML, reply_markup=markup)

    async def handle_sign(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return
        if self._throttle(query.from_user.id):
            await query.answer("Slow down")
            return
        parts = query.data.split(":")
        _, _, choice, game_id = parts
        game = self.mongo.fetch_game(game_id)
        if not game or game.get("status") != "SIGN_PICK":
            await query.answer("Game not available.", show_alert=True)
            return
        players = game.get("players", [])
        if query.from_user.id not in players:
            await query.answer("Only players can pick.", show_alert=True)
            return
        if game.get("playerX") or game.get("playerO"):
            await query.answer("Sign already chosen.", show_alert=True)
            return
        p1, p2 = players[0], players[1]
        if choice == "r":
            choice = "x" if uuid.uuid4().int % 2 == 0 else "o"
        if choice == "x":
            player_x, player_o = p1, p2
        else:
            player_x, player_o = p2, p1
        updated = self.mongo.update_game(
            game_id,
            {"$set": {"playerX": player_x, "playerO": player_o, "turn": "X", "status": "ACTIVE"}},
        )
        await query.answer("Signs locked. Game start!")
        await self.start_board(updated, context)
        await self.log_event(game["groupId"], game_id, "sign_selected", {"playerX": player_x, "playerO": player_o}, context)

    async def start_board(self, game: Dict, context: ContextTypes.DEFAULT_TYPE) -> None:
        board = game.get("board", ["" for _ in range(9)])
        text, markup = board_markup(board, game["_id"])
        caption = build_card(
            "Game started",
            [f"❌ {game['playerX']}  vs  0️⃣ {game['playerO']}", "Turn: ❌"],
            "Tap a cell to move",
        )
        msg = await context.bot.send_message(game["groupId"], f"{caption}\n{text}", parse_mode=ParseMode.HTML, reply_markup=markup)
        self.mongo.update_game(game["_id"], {"$set": {"messageRefs.gameMsgId": msg.message_id}})

    async def handle_cell(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return
        if self._throttle(query.from_user.id):
            await query.answer("Hold on")
            return
        parts = query.data.split(":")
        if len(parts) < 4:
            await query.answer("Malformed move", show_alert=True)
            return
        idx = int(parts[2])
        game_id = parts[3]
        game = self.mongo.fetch_game(game_id)
        if not game or game.get("status") != "ACTIVE":
            await query.answer("Game finished.", show_alert=True)
            return
        group = self.mongo.groups.find_one({"_id": game["groupId"]})
        if group and group.get("activeGameId") != game_id:
            await query.answer("Stale game.", show_alert=True)
            return
        player_symbol = "X" if query.from_user.id == game.get("playerX") else "O" if query.from_user.id == game.get("playerO") else None
        if not player_symbol:
            await query.answer("You are not in this game.", show_alert=True)
            return
        if game.get("turn") != player_symbol:
            await query.answer("Not your turn.", show_alert=True)
            return
        board = game.get("board", ["" for _ in range(9)])
        if board[idx]:
            await query.answer("Cell already taken", show_alert=True)
            return
        board[idx] = player_symbol
        move = {"by": query.from_user.id, "symbol": player_symbol, "idx": idx, "ts": dt.datetime.utcnow()}
        self.mongo.update_game(
            game_id,
            {"$set": {"board": board, "turn": "O" if player_symbol == "X" else "X", "updatedAt": dt.datetime.utcnow()}, "$push": {"moves": move}},
        )
        winner, line = check_winner(board)
        chat_id = game["groupId"]
        msg_id = game.get("messageRefs", {}).get("gameMsgId")
        text, markup = board_markup(board, game_id, highlight=line or [])
        next_turn = "O" if player_symbol == "X" else "X"
        status_lines = [f"❌ {game['playerX']} vs 0️⃣ {game['playerO']}"]
        if winner == "DRAW":
            status_lines.append("Result: Draw")
        elif winner:
            winner_id = game["playerX"] if winner == "X" else game["playerO"]
            status_lines.append(f"Winner: {winner_id}")
        else:
            turn_user = game["playerX"] if next_turn == "X" else game.get("playerO")
            status_lines.append(f"Turn: {turn_user}")
        caption = build_card("TicToe", status_lines)
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=f"{caption}\n{text}",
                parse_mode=ParseMode.HTML,
                reply_markup=None if winner else markup,
            )
        except Exception:
            await context.bot.send_message(chat_id, f"{caption}\n{text}", parse_mode=ParseMode.HTML, reply_markup=None if winner else markup)
        if winner:
            result_type = "DRAW" if winner == "DRAW" else "WIN"
            winner_id = None if winner == "DRAW" else (game["playerX"] if winner == "X" else game["playerO"])
            loser_id = None if winner == "DRAW" else (game["playerO"] if winner == "X" else game["playerX"])
            self.mongo.record_result(game_id, {"type": result_type, "winnerUserId": winner_id, "line": line})
            self.mongo.clear_active_game_if_match(chat_id, game_id)
            self.mongo.bump_stats(winner_id, loser_id, draw=winner == "DRAW")
            await self.log_event(chat_id, game_id, "game_ended", {"winner": winner_id, "draw": winner == "DRAW"}, context)
            await context.bot.send_message(
                chat_id,
                build_card(
                    "Game over",
                    [f"Winner: {winner_id}" if winner_id else "Draw!"],
                    "Use 🔄 Rematch or 🏁 New Game",
                ),
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("🔄 Rematch", callback_data=f"ttt:rematch:{game_id}")],
                        [InlineKeyboardButton("🏁 New Game", callback_data=f"ttt:new:{game_id}")],
                    ]
                ),
            )
        else:
            await query.answer()

    async def flee(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        game_id = None
        if query and query.data:
            parts = query.data.split(":")
            if len(parts) >= 3:
                game_id = parts[2]
        if update.effective_chat:
            game = self.current_game(update.effective_chat.id)
        else:
            game = None
        if game_id and (not game or game["_id"] != game_id):
            game = self.mongo.fetch_game(game_id)
        if not game:
            if query:
                await query.answer("No active game")
            return
        if game.get("status") == "ACTIVE":
            opponent = None
            if query and query.from_user.id == game.get("playerX"):
                opponent = game.get("playerO")
            elif query and query.from_user.id == game.get("playerO"):
                opponent = game.get("playerX")
            result = {"type": "FORFEIT", "winnerUserId": opponent, "line": None}
            self.mongo.record_result(game["_id"], result)
            self.mongo.bump_stats(opponent, query.from_user.id if query else None, draw=False)
        self.mongo.clear_active_game_if_match(game["groupId"], game["_id"])
        if query:
            await query.answer("Game ended")
        await context.bot.send_message(
            game["groupId"],
            build_card("Game cancelled", ["Lobby closed or player fled."], "Use /startgame to play again."),
        )
        await self.log_event(game["groupId"], game["_id"], "flee", {"by": query.from_user.id if query else None}, context)

    async def forceend(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.effective_user:
            return
        member = await update.effective_chat.get_member(update.effective_user.id)
        if member.status not in {"administrator", "creator"} and not is_owner(update.effective_user.id, self.settings):
            await update.message.reply_text("Admins only.")
            return
        game = self.current_game(update.effective_chat.id)
        if not game:
            await update.message.reply_text("No active game.")
            return
        self.mongo.clear_active_game_if_match(update.effective_chat.id, game["_id"])
        await update.message.reply_text("Game forcibly ended.")
        await self.log_event(update.effective_chat.id, game["_id"], "forceend", {"by": update.effective_user.id}, context)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user:
            return
        doc = self.mongo.user_stats(update.effective_user.id) or {}
        stats = doc.get("stats") or {"wins": 0, "losses": 0, "draws": 0, "gamesPlayed": 0}
        body = [
            f"Wins: {stats.get('wins',0)}",
            f"Losses: {stats.get('losses',0)}",
            f"Draws: {stats.get('draws',0)}",
            f"Games: {stats.get('gamesPlayed',0)}",
        ]
        await update.effective_message.reply_html(build_card("Your stats", body))

    async def groupstats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat:
            return
        stats = self.mongo.group_stats(update.effective_chat.id)
        body = [
            f"Total games: {stats['total']}",
            f"Wins: {stats['wins']}",
            f"Draws: {stats['draws']}",
            f"Losses: {stats['losses']}",
        ]
        await update.effective_message.reply_html(build_card("Group stats", body))

    async def broadcast(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not is_owner(update.effective_user.id, self.settings):
            await update.message.reply_text("Owner only")
            return
        if not context.args:
            await update.message.reply_text("Usage: /broadcast <message>")
            return
        msg = " ".join(context.args)
        successes = 0
        for gid in self.mongo.all_group_ids():
            try:
                await context.bot.send_message(gid, msg)
                successes += 1
                await asyncio.sleep(0.25)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Broadcast to %s failed: %s", gid, exc)
        await update.message.reply_text(f"Broadcast delivered to {successes} chats")

    async def setloggroup(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.effective_user:
            return
        member = await update.effective_chat.get_member(update.effective_user.id)
        if member.status not in {"administrator", "creator"} and not is_owner(update.effective_user.id, self.settings):
            await update.message.reply_text("Admins only.")
            return
        if not context.args:
            await update.message.reply_text("Usage: /setloggroup <chat_id|null>")
            return
        arg = context.args[0]
        log_chat_id = None if arg.lower() == "null" else int(arg)
        self.mongo.set_log_group(update.effective_chat.id, log_chat_id)
        await update.message.reply_text("Log destination updated")

    async def accept_invite(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return
        if self._throttle(query.from_user.id):
            await query.answer("Slow")
            return
        parts = query.data.split(":")
        game_id = parts[2]
        game = self.mongo.fetch_game(game_id)
        if not game or game.get("status") != "LOBBY":
            await query.answer("Lobby expired", show_alert=True)
            return
        invited = game.get("invitedUser")
        if invited != query.from_user.id:
            await query.answer("Not your invite", show_alert=True)
            return
        players = game.get("players", [])
        if len(players) >= 2:
            await query.answer("Lobby full", show_alert=True)
            return
        players.append(query.from_user.id)
        updated = self.mongo.update_game(game_id, {"$set": {"players": players, "status": "SIGN_PICK"}})
        await query.answer("Joined! Pick a sign")
        await self.prompt_sign_choice(update, context, updated)
        await self.log_event(game["groupId"], game_id, "invite_accepted", {"by": query.from_user.id}, context)

    async def decline_invite(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return
        if self._throttle(query.from_user.id):
            await query.answer("Slow")
            return
        parts = query.data.split(":")
        game_id = parts[2]
        game = self.mongo.fetch_game(game_id)
        if not game:
            await query.answer("Lobby gone", show_alert=True)
            return
        await query.answer("Declined")
        await context.bot.send_message(game["groupId"], "Invitation declined.")
        await self.log_event(game["groupId"], game_id, "invite_declined", {"by": query.from_user.id}, context)

    async def rematch(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return
        parts = query.data.split(":")
        game_id = parts[2]
        game = self.mongo.fetch_game(game_id)
        if not game:
            await query.answer("Game missing", show_alert=True)
            return
        players = game.get("players", [])
        if query.from_user.id not in players:
            await query.answer("Players only", show_alert=True)
            return
        group_id = game["groupId"]
        if self.mongo.groups.find_one({"_id": group_id}).get("activeGameId"):
            await query.answer("Another game active", show_alert=True)
            return
        new_id = str(uuid.uuid4())
        now = dt.datetime.utcnow()
        new_game = {
            "_id": new_id,
            "groupId": group_id,
            "status": "SIGN_PICK",
            "createdAt": now,
            "updatedAt": now,
            "createdBy": query.from_user.id,
            "joinMode": "open",
            "invitedUser": None,
            "joinExpiresAt": None,
            "playerX": None,
            "playerO": None,
            "players": players,
            "board": ["" for _ in range(9)],
            "turn": None,
            "moves": [],
            "messageRefs": {},
            "result": None,
        }
        self.mongo.create_game(new_game)
        if not self.mongo.set_active_game_if_free(group_id, new_id):
            await query.answer("Could not lock rematch", show_alert=True)
            return
        await query.answer("Rematch! Pick signs")
        await self.prompt_sign_choice(update, context, new_game)

    async def new_game_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data or not query.message or not query.message.chat:
            return
        await query.answer()
        await context.bot.send_message(query.message.chat_id, "Use /startgame to begin a new lobby.")

    async def fallback_join_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return
        fake_update = Update(update.update_id, message=query.message)
        await self.join(fake_update, context)

    def build_app(self) -> Application:
        app = (
            Application.builder()
            .token(self.settings.bot_token)
            .rate_limiter(AIORateLimiter())
            .build()
        )
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("startgame", self.startgame))
        app.add_handler(CommandHandler("join", self.join))
        app.add_handler(CommandHandler("stats", self.stats))
        app.add_handler(CommandHandler("groupstats", self.groupstats))
        app.add_handler(CommandHandler("broadcast", self.broadcast))
        app.add_handler(CommandHandler("forceend", self.forceend))
        app.add_handler(CommandHandler("setloggroup", self.setloggroup))
        app.add_handler(CallbackQueryHandler(self.handle_sign, pattern=r"^ttt:sign:"))
        app.add_handler(CallbackQueryHandler(self.handle_cell, pattern=r"^ttt:cell:"))
        app.add_handler(CallbackQueryHandler(self.accept_invite, pattern=r"^ttt:accept:"))
        app.add_handler(CallbackQueryHandler(self.decline_invite, pattern=r"^ttt:decline:"))
        app.add_handler(CallbackQueryHandler(self.flee, pattern=r"^ttt:flee:"))
        app.add_handler(CallbackQueryHandler(self.rematch, pattern=r"^ttt:rematch:"))
        app.add_handler(CallbackQueryHandler(self.new_game_button, pattern=r"^ttt:new:"))
        app.add_handler(CallbackQueryHandler(self.fallback_join_button, pattern=r"^ttt:join:"))
        return app


async def main() -> None:
    settings = load_settings()
    bot = TicToeBot(settings)
    app = bot.build_app()
    logger.info("Starting polling bot")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    await app.updater.idle()


if __name__ == "__main__":
    asyncio.run(main())
