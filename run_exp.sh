for mode in on off; do
  if [ "$mode" = "on" ]; then
    EXTRA="enable_correction=True|enable_highway_gate=True"
  else
    EXTRA="enable_correction=False|enable_highway_gate=False"
  fi

  # 修改后的 train.py 命令
  CUDA_VISIBLE_DEVICES=1 python train.py \
    -mc cfg/model/linear_tf.yaml \
    -dc cfg/data/omniglot.yaml \
    -l runs/quick_gate_cmp_${mode} \
    -o "tasks=5|train_shots=1|test_shots=1|max_train_steps=200|eval_interval=200|summary_interval=200|ckpt_interval=200|eval_iters=4|batch_size=256|eval_batch_size=64|tf_layers=1|hidden_dim=128|tf_heads=4|tf_ff_dim=256|attn_loss=0.0|distributed_loss=False|num_workers=16|${EXTRA}"

  # 修改后的 meta_train_score.py 命令
  CUDA_VISIBLE_DEVICES=1 python meta_train_score.py \
    -mc cfg/model/linear_tf.yaml \
    -dc cfg/data/omniglot.yaml \
    -l runs/quick_gate_cmp_${mode} \
    -l runs/quick_gate_cmp_${mode} \
    -o "tasks=5|train_shots=1|test_shots=1|max_train_steps=200|eval_iters=8|batch_size=256|eval_batch_size=64|tf_layers=1|hidden_dim=128|tf_heads=4|tf_ff_dim=256|attn_loss=0.0|distributed_loss=False|num_workers=16|${EXTRA}"
done

python - <<'PY'
import torch
on = torch.load('runs/quick_gate_cmp_on/meta_train_scores.pt')
off = torch.load('runs/quick_gate_cmp_off/meta_train_scores.pt')
print('ON  acc/train =', on.get('acc/train'))
print('OFF acc/train =', off.get('acc/train'))
print('DELTA         =', (on.get('acc/train',0)-off.get('acc/train',0)))
PY