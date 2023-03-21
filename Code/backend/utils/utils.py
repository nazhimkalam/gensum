from cryptography.fernet import Fernet


def encrypt_data(data, key):
    """
    Encrypts the given data using the provided secret key.
    Returns the encrypted data as a bytes object.
    """
    f = Fernet(key)
    return f.encrypt(data.encode())

def decrypt_data(encrypted_data, key):
    """
    Decrypts the given encrypted data using the provided secret key.
    Returns the decrypted data as a string.
    """
    f = Fernet(key)
    decrypted_bytes = f.decrypt(encrypted_data)
    return decrypted_bytes.decode()
