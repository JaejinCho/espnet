# This is generated from "rsync -av --exclude-from=temp /export/b10/jcho/espnet3/egs/libritts/asrtts+spkid/ asrtts+spkid/" at ../
# Then, symbolic links are generated for /export/b10/jcho/espnet3/egs/libritts/asrtts+spkid/{data,dump,exp,log}

- This asrtts+spkid/ directory is created to run same tts+spkid experiments
  using asr transcript:
  - Major changes:
    - run.ttsspkid.spkloss_weight.new.sh > run.asrttsspkid.spkloss_weight.new.sh

- Useful utility function:
    - print_losses: print validation losses ordered with corresponding epochs
    e.g) bash print_losses.sh exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.03

- Change in codes related to adding spkconsistency_loss
