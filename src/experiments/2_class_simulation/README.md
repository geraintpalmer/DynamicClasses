# 2-class simulations

This directory contains the scripts necessary to generate the 2-class simulations dataset. The `write_row_simulation.py` script is used to generate a single row of the dataset where the parameters are given as arguments to the script. The `generate_commands_simulation.py` script is used to generate the `commands.txt` file which contains all the commands necessary to generate the dataset `all_simulations.csv`. To generate the dataset using `60` cores, run the following commands:

```bash
python generate_commands_simulation.py
parallel --jobs 60 < commands.txt
```
