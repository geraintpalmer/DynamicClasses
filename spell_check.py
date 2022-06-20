import pathlib
import subprocess

import sys


def main(path, known_words):
    latex = path.read_text()
    aspell_output = subprocess.check_output(
        ["aspell", "-t", "--list", "--lang=en_GB"], input=latex, text=True
    )
    incorrect_words = set(aspell_output.split("\n")) - {""} - known_words
    if len(incorrect_words) > 0:
        exit_code = 1
        print(f"In {path} the following words are not known: ")
        for string in sorted(incorrect_words):
            print(string)
        sys.exit(exit_code)


if __name__ == "__main__":
    path = pathlib.Path("./tex/main.tex")
    known_words = {
        "TODO",
        "Markovian",
        "img",
        "pathreplacing",
    }
    main(path=path, known_words=known_words)
