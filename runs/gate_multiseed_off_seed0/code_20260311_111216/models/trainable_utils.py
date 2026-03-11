def set_trainable_modules(model, train_backbone=False, train_corr=True, train_gate=True, train_head=True):
    if hasattr(model, 'set_trainable_modules'):
        model.set_trainable_modules(
            train_backbone=train_backbone,
            train_corr=train_corr,
            train_gate=train_gate,
            train_head=train_head,
        )
    return model