def set_trainable_modules(
        model,
        train_backbone=False,
        train_corr=True,
        train_gate=True,
        train_head=True,
        partial_freeze_mode='none',
        train_last_tf_layers=0,
        freeze_encoder=False):
    if hasattr(model, 'set_trainable_modules'):
        model.set_trainable_modules(
            train_backbone=train_backbone,
            train_corr=train_corr,
            train_gate=train_gate,
            train_head=train_head,
            partial_freeze_mode=partial_freeze_mode,
            train_last_tf_layers=train_last_tf_layers,
            freeze_encoder=freeze_encoder,
        )
    return model