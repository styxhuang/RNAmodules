## Install
- clone package to your folder
- `pip install -e .` install rna_tool package

## usage
- `rna -h` shows the detail
- `rna -i PATH\TO\config.yaml -o PATH\TO\RESULTS -t -p -v` to run your fasta file
- "config.yaml" examples can find under tests/assets_small_train_data/config.yaml

## NOTE
- The current time tracking utility has an issue: time spent in loops is being counted multiple times. This should be fixed in the future.