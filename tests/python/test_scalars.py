from penumbra_fhe import (
    SecurityParams,
    decrypt,
    encrypt,
    keygen,
    set_server_key,
)


def test_encrypted_add() -> None:
    params = SecurityParams(42)
    client_key, server_key = keygen(params)
    set_server_key(server_key)

    x = 10
    y = 25

    ct_x = encrypt(x, client_key)
    ct_y = encrypt(y, client_key)

    ct_res = ct_x + ct_y
    res = decrypt(ct_res, client_key)

    assert res == x + y


def test_encrypted_sub() -> None:
    params = SecurityParams(42)
    client_key, server_key = keygen(params)
    set_server_key(server_key)

    x = 100
    y = 25

    ct_x = encrypt(x, client_key)
    ct_y = encrypt(y, client_key)

    ct_res = ct_x - ct_y
    res = decrypt(ct_res, client_key)

    assert res == x - y


def test_encrypted_mul() -> None:
    params = SecurityParams(42)
    client_key, server_key = keygen(params)
    set_server_key(server_key)

    x = 10
    k = 5

    ct_x = encrypt(x, client_key)

    ct_res = ct_x * k
    res = decrypt(ct_res, client_key)

    assert res == x * k
