from box import Box

num_frames = 4
num_frames_test = 4
batch_size = 8
batch_size_test = 64
max_txt_l = 512


config = Box(
    {
        "model": Box(
            dict(
                model_cls="InternVideo2_Stage2",
                vision_encoder=Box(
                    dict(
                        # backbone
                        name="pretrain_internvideo2_1b_patch14_224",
                        img_size=224,
                        num_frames=num_frames,
                        tubelet_size=1,
                        patch_size=14,
                        d_model=1408,
                        clip_embed_dim=768,
                        clip_teacher_embed_dim=3200,
                        clip_teacher_final_dim=768,
                        clip_norm_type="l2",
                        clip_return_layer=6,
                        clip_student_return_interval=1,
                        pretrained="data/models/InternVideo2/InternVideo2-stage2_1b-224p-f4.pt",
                        use_checkpoint=True,
                        checkpoint_num=40,
                        use_flash_attn=False,
                        use_fused_rmsnorm=False,
                        use_fused_mlp=False,
                        # clip teacher
                        clip_teacher=None,
                        clip_input_resolution=224,
                        clip_teacher_return_interval=1,
                        # mask
                        video_mask_type="random",
                        video_mask_ratio=0.8,
                        image_mask_type="random",
                        image_mask_ratio=0.5,
                        sep_image_video_pos_embed=True,
                        keep_temporal=False,
                        only_mask=True,
                    )
                ),
                text_encoder=Box(
                    dict(
                        name="bert_large",
                        pretrained="bert-large-uncased",
                        config="src/eval/InternVideo2/config_bert_large.json",
                        d_model=1024,
                        fusion_layer=19,
                    )
                ),
                multimodal=Box(dict(enable=True)),
                embed_dim=512,
                temp=0.07,
                find_unused_parameters=False,
            )
        ),
        "inputs": Box(
            dict(
                image_res=224,
                video_input=Box(
                    dict(
                        num_frames=num_frames,
                        sample_type="rand",
                        num_frames_test=num_frames_test,
                        sample_type_test="middle",
                        random_aug=False,
                    )
                ),
                max_txt_l=Box(dict(image=max_txt_l, video=max_txt_l)),
                batch_size=Box(dict(image=batch_size, video=batch_size)),
                batch_size_test=Box(dict(image=batch_size_test, video=batch_size_test)),
            )
        ),
        "evaluation": Box(
            dict(
                eval_frame_ensemble="concat",  # [concat, max, mean, lse]
                eval_x_only=False,
                k_test=128,
                eval_offload=True,  # offload gpu tensors to cpu to save memory.
            )
        ),
        "gradient_checkpointing": True,
        "use_flash_sdp": False,
        "use_half_precision": False,
        "max_txt_l": max_txt_l,
        "criterion": Box(
            dict(
                loss_weight=dict(
                    vtc=1.0,
                    mlm=1.0,
                    vtm=1.0,
                    mvm=0.0,
                    uta=0.0,
                ),  # 0: disabled.
                vtm_hard_neg=True,
                mlm_masking_prob=0.5,
                distill_final_features=True,
                clip_loss_ratio=[1.0, 1.0],
            )
        ),
        "optimizer": Box(
            dict(
                opt="adamW",
                lr=1e-5,
                opt_betas=[0.9, 0.98],  # default
                weight_decay=0.05,
                max_grad_norm=3.0,  # requires a positive float, use -1 to disable
                # use a different lr for some modules, e.g., larger lr for new modules
                different_lr=dict(enable=False, module_names=[], lr=1e-3),
            )
        ),
        "scheduler": Box(dict(sched="cosine", epochs=1, min_lr_multi=0.01, warmup_epochs=0.2)),
        "pretrained_path": "data/models/InternVideo2/InternVideo2-stage2_1b-224p-f4.pt",
    }
)
