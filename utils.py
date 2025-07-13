from typing import Any, Dict, List

RATING_TO_SENTIMENT = {
    1.0: 'negative',
    2.0: 'negative',
    3.0: 'neutral',
    4.0: 'positive',
    5.0: 'positive',
}

LABELS = [
    'positive',
    'neutral',
    'negative',
]


def print_formatted_dictionaries(list_of_dicts: List[Dict[Any, Any]]) -> None:
    """
    Iterates through a list of dictionaries and prints each one in a readable format.

    This function follows PEP 8 styling, includes type hints, and adds
    clear separators between each dictionary for better readability.

    Args:
        list_of_dicts (List[Dict[Any, Any]]): A list containing one or more
                                             dictionary objects.
    """
    if not isinstance(list_of_dicts, list):
        print("Error: Input must be a list of dictionaries.")
        return

    for index, dictionary in enumerate(list_of_dicts):
        # Create a header for each dictionary for clear separation
        header = f"--- Dictionary {index + 1} ---"
        print(header)

        if not isinstance(dictionary, dict):
            print("  Warning: Item in list is not a dictionary.")
            print("-" * len(header))
            continue

        if not dictionary:
            print("  (This dictionary is empty)")

        for key, value in dictionary.items():
            print(f"  {str(key)}: {str(value)}")

        # Print a closing separator line
        print("-" * len(header))


'''
# --- Example Usage ---
# Call the function with the sample data
print_formatted_dictionaries(sample_data)
'''
