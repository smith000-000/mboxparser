from email.message import EmailMessage

from mbox_thread_dump import extract_addresses


def test_extract_addresses_display_name():
    msg = EmailMessage()
    msg["From"] = "Jane Doe <JANE@EXAMPLE.COM>"

    assert extract_addresses(msg, include_cc=False) == {"jane@example.com"}


def test_extract_addresses_multiple_recipients():
    msg = EmailMessage()
    msg["To"] = "a@example.com, B User <b@EXAMPLE.com>"
    msg["Cc"] = "c@example.com"

    assert extract_addresses(msg, include_cc=True) == {
        "a@example.com",
        "b@example.com",
        "c@example.com",
    }


def test_extract_addresses_weird_whitespace():
    msg = EmailMessage()
    msg["To"] = "  Alice  <alice@example.com> ,   bob@example.com  "

    assert extract_addresses(msg, include_cc=False) == {
        "alice@example.com",
        "bob@example.com",
    }


def test_extract_addresses_uppercase_and_cc_excluded():
    msg = EmailMessage()
    msg["From"] = "SENDER@EXAMPLE.COM"
    msg["Cc"] = "CC@EXAMPLE.COM"

    assert extract_addresses(msg, include_cc=False) == {"sender@example.com"}
