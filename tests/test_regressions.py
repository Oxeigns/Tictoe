import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import TicToeBot, mention_html
from mongodb import Mongo


class DatabaseTests(unittest.TestCase):
    def test_explicit_database_does_not_use_truth_value(self):
        client = MagicMock()
        database = MagicMock()
        database.__bool__.side_effect = NotImplementedError
        client.get_default_database.return_value = database
        with patch('mongodb.MongoClient', return_value=client):
            result = Mongo(SimpleNamespace(mongo_url='mongodb://localhost/test'))
        self.assertIs(result.db, database)

    def test_html_names_are_escaped(self):
        self.assertIn('&lt;name&gt; &amp;', mention_html(1, '<name> &'))


class ShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_expiry_task_does_not_escape(self):
        bot = TicToeBot.__new__(TicToeBot)
        bot._expiry_task = asyncio.create_task(asyncio.sleep(3600))
        await bot.post_shutdown(None)
        self.assertTrue(bot._expiry_task.cancelled())
