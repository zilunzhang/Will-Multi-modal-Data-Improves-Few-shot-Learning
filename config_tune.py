import ray.tune as tune


def set_up_config(select_func=tune.grid_search):
    config = {
            "seed":
                select_func([
                    # 222,
                    0,
                    # 999,
                    # 77,
                    # 666,
                    # 28,
                ]),
            "backbone":
                select_func([
                    'ConvNet',
                    # 'ResNet12',
                    # 'resnet18',
                ]),

            "model":
                select_func([
                    'ProtoNet',
                    # 'MAML',
                ]),
            "emb_size":
                select_func([
                    # 1600,
                    128,
                    # 800,
                ]),

            "query_type":
                select_func([
                    'uniform',
                ]),

            "lr":
                select_func([
                    1e-3,
                ]),
            "weight_decay":
                select_func([
                    1e-3,
                ]),
            "lr_schedule_step_size":
                select_func([
                        # 20,
                        # 50,
                        80,
                ]),
            "lr_schedule_gamma":
                select_func([
                    0.5,
                    # 0.3,
                    # 0.7,
                    # 0.9,
                ]),
        }

    return config
