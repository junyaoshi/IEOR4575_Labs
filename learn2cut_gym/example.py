from gymenv_v2 import make_multiple_env
import numpy as np
from tqdm import trange
from learntocut.policy import MLPRowFeatureAttenttionPolicy
from learntocut.optimizer import Adam
import os


import wandb
wandb.login()
model = 'random'  # 'attention'
difficulty = 'hard'
testing = True
mode_tag = f'{difficulty}-{model}'
trial_tag = 'official'
official_tag = f"{'testing' if testing else 'training'}-{difficulty}"
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-easy"])
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-hard"])
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["baby-random"])
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["easy-random"])
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["hard-random"])
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["baby-attention"])
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["baby-attentionembed"])
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["baby-lstmembed"])
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["easy-attention"])
run=wandb.init(project="finalproject", entity="ieor-4575", tags=[mode_tag, trial_tag, official_tag])


learning_rate = 0.01
delta_std = 1  # 0.02
gamma = 0.95
n_episodes = 50

### TRAINING

# Setup: You may generate your own instances on which you train the cutting agent.
custom_config = {
    "load_dir"        : 'instances/randomip_n60_m60',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(20)),                # take the first 20 instances from the directory
    "timelimit"       : 50,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

# Baby Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
baby_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : [0],
    "timelimit"       : 100,
    "reward_type"     : 'obj'
}

# Easy Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
easy_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 100,
    "reward_type"     : 'obj'
}

# Hard Setup: Use the following environment settings. We will evaluate your agent with the same hard config below:
hard_config = {
    "load_dir"        : 'instances/train_100_n60_m60',
    "idx_list"        : list(range(99)),
    "timelimit"       : 100,
    "reward_type"     : 'obj'
}

# Test Setup: testing generalization
test_config = {
"load_dir" : 'instances/test_100_n60_m60',
"idx_list" : list(range(99)),
"timelimit" : 100,
"reward_type" : 'obj'
}

if __name__ == "__main__":
    # create env
    if testing:
        env = make_multiple_env(**test_config)
    elif difficulty == 'easy':
        env = make_multiple_env(**easy_config)
    elif difficulty == 'hard':
        env = make_multiple_env(**hard_config)
    elif difficulty == 'baby':
        env = make_multiple_env(**baby_config)
    else:
        raise ValueError("Unrecognized difficulty")

    rrecord = []
    discounted_rrecord = []
    trecord = []
    fixed_window = 8

    if model == 'random':
        policy = None
        optimizer = None
    elif model == 'attention':
        stats = {'numvars': 60}
        policy_param = {'numvars': stats['numvars'],
                        'ob_filter': 'NoFilter'}
        policy_param['hsize'] = 64
        policy_param['numlayers'] = 2
        policy_param['embed'] = 32
        policy_param['rowembed'] = 32
        policy = MLPRowFeatureAttenttionPolicy(policy_param)
        optimizer = Adam(policy.get_weights(), learning_rate)
    else:
        raise ValueError(f'Unrecognized model: {model}')

    if testing and model != 'random':
        # load data
        logdir = os.path.join('models', mode_tag)
        params, mu, std = np.load(logdir + '/params.npy', allow_pickle=True)
        policy.update_weights(params)
        print('setting filter')
        if hasattr(policy.observation_filter, 'mu'):
            policy.observation_filter.mu = mu
        if hasattr(policy.observation_filter, 'std'):
            policy.observation_filter.mu = std
        adam_v = np.load(logdir + '/adam_v.npy', allow_pickle=True)
        adam_m = np.load(logdir + '/adam_m.npy', allow_pickle=True)
        optimizer.v = adam_v
        optimizer.m = adam_m
        print(f'Model Loaded from {logdir}')

    for e in trange(n_episodes):
        # gym loop
        s = env.reset()   # samples a random instance every time env.reset() is called
        d = False
        t = 0
        discounted_r = 0.
        repisode = 0.

        if model == 'random':
            factor = 1.0
            while not d:
                a = np.random.randint(0, s[-1].size, 1)
                # s[-1].size shows the number of actions, i.e., cuts available at state s
                s, r, d, _ = env.step(list(a))
                # print(f'episode: {e} | step: {t} | reward: {r} | aspace size: {s[-1].size} | action: {a[0]}')

                A, b, c0, cuts_a, cuts_b = s
                #print(A.shape, b.shape, c0.shape, cuts_a.shape, cuts_b.shape)

                discounted_r += r * factor
                repisode += r
                factor *= gamma
                t += 1
            rrecord.append(repisode)
            discounted_rrecord.append(discounted_r)
            trecord.append(t)
        else:
            if testing:
                factor = 1.0
                while not d:
                    a = policy.act(s)
                    s, r, d, _ = env.step([a])
                    A, b, c0, cuts_a, cuts_b = s

                    discounted_r += r * factor
                    repisode += r
                    factor *= gamma
                    t += 1
                rrecord.append(repisode)
                discounted_rrecord.append(discounted_r)
                trecord.append(t)
            else:
                original_weights = policy.get_weights().copy()

                # keep record of training rewards and epsilon
                epsilon_table = []
                train_rewards_table = []

                # sample n_directions noise
                epsilon = np.random.randn(*policy.get_weights().shape) * delta_std

                # update weights
                policy.update_weights(original_weights + epsilon)

                # estimate gradients
                rewards = []
                times = []

                factor = 1.0

                while not d:
                    a = policy.act(s)
                    s, r, d, _ = env.step([a])
                    # print(f'episode: {e} | step: {t} | reward: {r} | '
                    #       f'aspace size: {s[-1].size} | action: {a}')
                    A, b, c0, cuts_a, cuts_b = s

                    discounted_r += r * factor
                    repisode += r
                    factor *= gamma
                    t += 1

                rrecord.append(repisode)
                discounted_rrecord.append(discounted_r)
                trecord.append(t)

                rewards.append(discounted_r)
                times.append(t)

                # record rewards and epsilon
                epsilon_table.append(epsilon)
                train_rewards_table.append(np.mean(rewards))

                # acumulate gradients
                epsilon_table = np.array(epsilon_table)
                train_rewards_table = np.array(train_rewards_table)
                # train_rewards_table = (train_rewards_table - np.mean(train_rewards_table)) / (
                #             np.std(train_rewards_table) + 1e-8)

                grad = np.mean(epsilon_table * train_rewards_table[:, np.newaxis], axis=0) / delta_std
                # print(f'Grad Norm: {np.linalg.norm(grad)}')

                # assign back the original params
                policy.update_weights(original_weights)

                # update
                w = policy.get_weights() - optimizer._compute_step(grad.flatten()).reshape(policy.get_weights().shape)
                policy.update_weights(w)

        # wandb logging
        movingAverage = 0
        if len(rrecord) >= fixed_window:
            movingAverage = np.mean(rrecord[len(rrecord) - fixed_window:len(rrecord)])
        discountedMovingAverage = 0
        if len(discounted_rrecord) >= fixed_window:
            discountedMovingAverage = np.mean(
                discounted_rrecord[len(discounted_rrecord) - fixed_window:len(discounted_rrecord)])
        ncutMovingAverage = 0
        if len(trecord) >= fixed_window:
            ncutMovingAverage = np.mean(trecord[len(trecord) - fixed_window:len(trecord)])
        wandb.log({"Episode reward": repisode,
                   "Episode reward moving average": movingAverage,
                   "Episode discounted reward": discounted_r,
                   "Episode discounted reward moving average": discountedMovingAverage,
                   "Number of cuts": t,
                   "Number of cuts moving average": ncutMovingAverage})
        print(f'End of episode {e} | '
              f'Episode Reward: {repisode} | '
              f'Episode Reward Moving Average: {movingAverage} | '
              f'Episode Discounted Reward: {discounted_r} | '
              f'Episode Discounted Reward Moving Average: {discountedMovingAverage} | '
              f'Number of Cuts: {t} | '
              f'Number of Cuts Moving Average: {ncutMovingAverage}')

    if not testing and model != 'random':
        # save everything
        logdir = os.path.join('models', mode_tag)
        np.save(logdir + '/params', policy.get_weights_plus_stats())
        np.save(logdir + '/adam_v', optimizer.v)
        np.save(logdir + '/adam_m', optimizer.m)
        np.save(logdir + '/rrecord', np.array(rrecord))
        np.save(logdir + '/discounted_rrecord', np.array(discounted_rrecord))
        np.save(logdir + '/trecord', np.array(trecord))
        print(f'Model Saved at {logdir}')

    # final logging
    print('Finished! Printing final report...')
    print(f'Average reward: {np.mean(rrecord)} | '
          f'Std reward: {np.std(rrecord)}')
    print(f'Average discounted reward: {np.mean(discounted_rrecord)} | '
          f'Std discounted reward: {np.std(discounted_rrecord)}')
    print(f'Average number of cuts: {np.mean(trecord)} | '
          f'Std number of cuts: {np.std(trecord)}')
