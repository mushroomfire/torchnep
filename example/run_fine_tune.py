from torchnep import train_nep

train_nep(
    "nep_finetune.in",
    "train_Si.xyz",
    output_dir="finetune_output",
    finetune_from="nep89.txt", 
    print_interval=1,
    slim_types=True
)