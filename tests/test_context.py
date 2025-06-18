from structured_logging.context import (
    get_custom_context,
    get_request_id,
    get_user_context,
    request_context,
    set_custom_context,
    set_request_id,
    set_user_context,
    update_custom_context,
)


def test_request_id():
    assert get_request_id() == ""

    set_request_id("test-123")
    assert get_request_id() == "test-123"

    set_request_id("")
    assert get_request_id() == ""


def test_user_context():
    assert get_user_context() == {}

    context = {"user_id": "user123", "tenant_id": "tenant456"}
    set_user_context(context)
    assert get_user_context() == context

    set_user_context({})
    assert get_user_context() == {}


def test_custom_context():
    assert get_custom_context() == {}

    context = {"service": "api", "version": "1.0"}
    set_custom_context(context)
    assert get_custom_context() == context

    update_custom_context(environment="prod")
    expected = {"service": "api", "version": "1.0", "environment": "prod"}
    assert get_custom_context() == expected


def test_request_context_manager():
    # Clean state before test
    set_request_id("")
    set_user_context({})
    set_custom_context({})

    with request_context(
        user_id="user123", tenant_id="tenant456", service="api"
    ) as req_id:
        assert len(req_id) == 36  # UUID length
        assert get_request_id() == req_id
        assert get_user_context() == {"user_id": "user123", "tenant_id": "tenant456"}
        assert get_custom_context() == {"service": "api"}

    assert get_request_id() == ""
    assert get_user_context() == {}
    assert get_custom_context() == {}


def test_request_context_with_optional_fields():
    with request_context(user_id="user123"):
        assert get_user_context() == {"user_id": "user123"}

    with request_context(tenant_id="tenant456"):
        assert get_user_context() == {"tenant_id": "tenant456"}

    with request_context():
        assert get_user_context() == {}
