import re


def _E0602_extract_chain(lead_identifier, line):
    results = []
    split_line = split_identifiers_non_identifiers(line)
    for idx, item in enumerate(split_line):
        if item == lead_identifier:
            for item2 in split_line[idx:]:
                if is_identifier(item2):
                    results.append(item2)
                else:
                    if item2.strip() == '.':
                        continue
                    else:
                        break
    return results


def E0602_handler(diagnostic_body):
    # search M-C, and
    # M-C
    # extract symbol
    symbol_list = extract_single_quoted_strings(diagnostic_body['message'])
    if len(symbol_list) != 1:
        raise AssertionError("E0602 pattern not match")
    symbol_names = _E0602_extract_chain(lead_identifier=symbol_list[0],
                                        line=diagnostic_body['line_content'])
    # generate report message
    message = f"In line: {diagnostic_body['line_content']}\nError: There is no symbol named '{symbol_names[0]}' in current context.\n"
    if diagnostic_body['module'].split('.')[-1] == symbol_names[0]:
        message += f"May not need to add module qualifier '{symbol_names[0]}'.\n"
    return message


def E1101_handler(diagnostic_body):  # no name in module
    # M_C_CF
    symbol_list = extract_single_quoted_strings(diagnostic_body['message'])
    if len(symbol_list) != 2 and len(symbol_list) != 3:
        raise AssertionError("E0602 pattern not match")
    class_name = symbol_list[0]
    symbol_name = symbol_list[1]
    message = f"In line: {diagnostic_body['line_content']}\nError: The class '{class_name}' has no member named '{symbol_name}'.\n"
    return message

    pass


def E0102_handler(diagnostic_body):  # function redefined
    message = f"In line: {diagnostic_body['line_content']}\nError: This function is already defined in previous context, you may directly use it."
    return message


def split_identifiers_non_identifiers(statement):
    # Splitting the string based on Python identifier rules
    # A Python identifier must start with a letter (a-z, A-Z) or underscore (_)
    # and can be followed by any number of letters, digits (0-9), or underscores
    identifier_pattern = r'\b[_a-zA-Z][_a-zA-Z0-9]*\b'
    parts = re.findall(identifier_pattern, statement)

    # Finding non-identifier parts
    non_identifier_parts = re.split(identifier_pattern, statement)

    # Combining identifier and non-identifier parts in order
    result = []
    for i in range(len(non_identifier_parts)):
        if non_identifier_parts[i]:
            result.append(non_identifier_parts[i])
        if i < len(parts):
            result.append(parts[i])

    return result


def is_identifier(string):
    if not string:  # Check if the string is empty
        return False

    if not (string[0].isalpha() or string[0] == '_'):  # Check if first character is a letter or underscore
        return False

    for char in string[1:]:  # Check remaining characters
        if not (char.isalnum() or char == '_'):
            return False

    return True


def extract_single_quoted_strings(input_string):
    # Regex pattern to match substrings within single quotes
    pattern = r"'([^']*)'"
    # Find all matches and return them as a list
    return re.findall(pattern, input_string)
