from collections import deque


def print_bin(bin: str) -> None:
    bytes = deque([])
    for i in range(len(bin), 0 , -4):
        bytes.appendleft(bin[max(0, i-4): i])
    print(" ".join(bytes))


def utf8_bin(char: str) -> str:
    binary = bin(ord(char))[2:]
    headers = ["", "0", "110", "1110", "11110"]
    bytes = deque([])
    for i in range(len(binary), 0 , -6):
        bytes.appendleft(binary[max(0, i-6): i])
    for i in range(1, len(bytes)):
        bytes[i] = "10" + bytes[i]
    bytes_len = len(bytes)
    assert bytes_len <= 4 and bytes_len > 0
    pad = 8 - (len(headers[bytes_len]) + len(bytes[0]))
    if pad > 0:
        bytes[0] = ("0" * pad) + bytes[0]
    bytes[0] = headers[bytes_len] + bytes[0]
    return " ".join(bytes)


def utf8_encode(char: str) -> bytes:
    binary = bin(ord(char))[2:]
    headers = ["", "0", "110", "1110", "11110"]
    bytes_str_list = deque([])
    for i in range(len(binary), 0 , -6):
        bytes_str_list.appendleft(binary[max(0, i-6): i])
    for i in range(1, len(bytes_str_list)):
        bytes_str_list[i] = "10" + bytes_str_list[i]
    bytes_str_list_len = len(bytes_str_list)
    assert bytes_str_list_len <= 4 and bytes_str_list_len > 0
    pad = 8 - (len(headers[bytes_str_list_len]) + len(bytes_str_list[0]))
    if pad > 0:
        bytes_str_list[0] = ("0" * pad) + bytes_str_list[0]
    bytes_str_list[0] = headers[bytes_str_list_len] + bytes_str_list[0]
    bytes_ints = [int(b, 2) for b in bytes_str_list]
    return bytes(bytes_ints)


def padd(string: str, to: int) -> str:
    str_len = len(string)
    pad = to - str_len
    return ("0" * pad) + string


def flatten(list_of_elements: list) -> list[bytes]:
    flat = []
    stack = [list_of_elements]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(reversed(cur))
        else:
            flat.append(cur)
    return flat


def utf8_decode(encoded: bytes) -> str:
    headers = ["", "0", "110", "1110", "11110"]
    bytes_ints = [b for b in encoded]
    bytes_str_list = [padd(str(bin(b))[2:], 8) for b in bytes_ints]
    stream_len = len(bytes_str_list)
    string = ""
    i = 0
    while i < stream_len:
        header_byte = bytes_str_list[i]
        num_of_bytes = 0
        for b in header_byte:
            if b != "1":
                break
            num_of_bytes += 1
        if num_of_bytes == 0: num_of_bytes = 1
        char_bytes = bytes_str_list[i:i+num_of_bytes]
        for j in range(len(char_bytes)):
            if j == 0:
                char_bytes[j] = char_bytes[j][len(headers[num_of_bytes]):]
            else:
                char_bytes[j] = char_bytes[j][2:]
        char_bin = "".join(char_bytes)
        char_val = int(char_bin, 2)
        char = chr(char_val)
        string += char
        i += num_of_bytes
    return string


def main() -> None:
    # upper_a = "A"
    # lower_a = chr((ord(upper_a) | 32))
    # print(ord(upper_a), upper_a)
    # print(ord(lower_a), lower_a)
    # a = 0x41
    # print(bytes([a]).decode(encoding="utf-8"))
    # print(int(0xffff))
    string = "中中"
    char_bytes = b"".join([utf8_encode(char) for char in string])
    print(char_bytes)
    print(utf8_decode(char_bytes))
    ...


if __name__ == "__main__":
    main()

