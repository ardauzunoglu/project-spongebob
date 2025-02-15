LRS=(0.001 0.0001 0.00001 0.000001)
EPOCHS=(10 25 50)
SEEDS=(42 300 5000)

for lr_val in "${LRS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python3 train_jp.py \
                --mode "regular" \
                --seed "$seed" \
                --data "Phi-3.5-mini-instructhidden_states.json" \
                --epochs "$epoch" \
                --lr "$lr_val" \

    done
  done
done