# Dynamic Classes

## Reproducing the results.

To create a virtual environment:

    $ python -m venv env

To start using the new virtual environment:

    $ source env/bin/activate

To install the dependencies:

    $ python -m pip install -r requirements.txt

To run the tests:

    $ cd src
    $ python -m pytest .

To check the format:

    $ python -m black src/

## Spelling checking

`check_spelling.py` contains a `known_words` set: add known words there.

This will run the spell checker:

    $ python check_spelling.py

Uses `aspell`.

## Check for insensitive language

Run:

    $ alex tex/main.tex

Uses `alex`.

To avoid a check, annotate using:

    % <!--alex disable-->
    Text with prose you want alex to ignore.
    % <!--alex enable-->
