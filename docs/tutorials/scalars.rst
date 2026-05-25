=================
Scalar Operations
=================

Penumbra-FHE supports fully homomorphic encryption of scalar 32-bit unsigned integers out of the box.

Initialization
==============

Before encrypting or executing operations, you must generate the client and server keys.

.. code-block:: python

    from penumbra_fhe import SecurityParams, keygen, set_server_key

    # Initialize security parameters with a deterministic PRNG seed
    params = SecurityParams(42)

    # Generate the client key (kept secret) and server key (can be public)
    client_key, server_key = keygen(params)

    # Set the server key in the current thread's context so that operators can access it
    set_server_key(server_key)


Encryption and Decryption
=========================

Use the ``encrypt`` and ``decrypt`` functions with your ``ClientKey``.

.. code-block:: python

    from penumbra_fhe import encrypt, decrypt

    x = 10

    # Encrypt the scalar
    ct_x = encrypt(x, client_key)

    # Decrypt the ciphertext back to plaintext
    res = decrypt(ct_x, client_key)

    assert res == x


Operations
==========

Once the server key is set, you can use standard Python operators to homomorphically compute on ciphertexts.

.. code-block:: python

    ct_y = encrypt(25, client_key)

    # Homomorphic addition
    ct_add = ct_x + ct_y

    # Homomorphic subtraction
    ct_sub = ct_x - ct_y

    # Homomorphic scalar multiplication (ciphertext * plaintext)
    ct_mul = ct_x * 5

    assert decrypt(ct_add, client_key) == 35
    assert decrypt(ct_sub, client_key) == -15 % (2**32)
    assert decrypt(ct_mul, client_key) == 50
