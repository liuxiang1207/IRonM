"""对抗题库的答题入口：用完整的对话链路回答（不调用模型，相当于模型全部不可用）。"""

import itertools

from ironm.orchestrator.chat import ChatReply, respond
from tests.memory_helpers import memory_db
from tests.test_chat_safety import _deps

_users = itertools.count(1)


def answer(prompt: str) -> ChatReply:
    """每道题用一个新用户，避免前面题目的画像影响后面的题。"""
    return respond(f"adv{next(_users)}", prompt, _deps([], memory_db()))


def reply_text(reply: ChatReply) -> str:
    return "\n".join([*(s.text for s in reply.sections), *(p.text for p in reply.answer)])
