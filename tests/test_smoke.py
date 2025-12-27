import mailbox
from email.message import EmailMessage

from mbox_thread_dump import build_thread_graph, expand_to_thread, find_seed_ids, parse_mbox


def _add_message(mbox, message_id, subject, from_addr, to_addr, references=None, in_reply_to=None):
    msg = EmailMessage()
    msg["Message-ID"] = message_id
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    if references:
        msg["References"] = references
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    msg.set_content("Hello")
    mbox.add(msg)


def test_thread_expansion_connected_component(tmp_path):
    mbox_path = tmp_path / "sample.mbox"
    mbox = mailbox.mbox(mbox_path)
    try:
        _add_message(
            mbox,
            "<id1@example.com>",
            "Seed",
            "target@example.com",
            "other@example.com",
        )
        _add_message(
            mbox,
            "<id2@example.com>",
            "Reply",
            "other@example.com",
            "target@example.com",
            references="<id1@example.com>",
        )
        _add_message(
            mbox,
            "<id3@example.com>",
            "Follow-up",
            "third@example.com",
            "other@example.com",
            references="<id1@example.com> <id2@example.com>",
            in_reply_to="<id2@example.com>",
        )
        mbox.flush()
    finally:
        mbox.close()

    parsed = parse_mbox(mbox_path)
    graph = build_thread_graph(parsed)
    seed_ids, _ = find_seed_ids(parsed["messages"], {"target@example.com"}, include_cc=False)
    selected_ids, _, thread_to_ids = expand_to_thread(seed_ids, graph)

    assert selected_ids == {"id1@example.com", "id2@example.com", "id3@example.com"}
    assert len(thread_to_ids) == 1
