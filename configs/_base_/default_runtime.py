checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/snamburu/siddhartha/actrecog/RoCoG_data_prep/i3d_normal/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth'
resume_from = None
workflow = [('train', 1)]
