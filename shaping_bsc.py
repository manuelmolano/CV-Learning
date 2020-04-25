#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 07:26:31 2020

@author: manuel

/home/.../shaping_run.py --folder test  --seed 4 --alg A2C --stages 3 4

"""
import os
from ops.utils import get_name_and_command_from_dict as gncfd
from ops.utils import rest_arg_parser
import sys
import numpy as np
# import numpy as np
import importlib
import argparse
sys.path.append(os.path.expanduser("~/gym"))
sys.path.append(os.path.expanduser("~/stable-baselines"))
sys.path.append(os.path.expanduser("~/neurogym"))
import gym
import neurogym as ngym  # need to import it so ngym envs are registered
from neurogym.utils import plotting
from neurogym.wrappers import ALL_WRAPPERS
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds


def test_env(env, kwargs, num_steps=100):
    """Test if all one environment can at least be run."""
    env = gym.make(env, **kwargs)
    env.reset()
    for stp in range(num_steps):
        action = 0
        state, rew, done, info = env.step(action)
        if done:
            env.reset()
    return env


def apply_wrapper(env, wrap_string, params):
    wrap_str = ALL_WRAPPERS[wrap_string]
    wrap_module = importlib.import_module(wrap_str.split(":")[0])
    wrap_method = getattr(wrap_module, wrap_str.split(":")[1])
    return wrap_method(env, **params)


def arg_parser():
    """
    Create an argparse.ArgumentParser for neuro environments
    """
    aadhf = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=aadhf)
    parser.add_argument('--dt', help='time step for environment', type=int,
                        default=None)
    parser.add_argument('--folder', help='where to save the data and model',
                        type=str, default=None)
    parser.add_argument('--alg', help='RL algorithm to use',
                        type=str, default=None)
    parser.add_argument('--task', help='task', type=str, default=None)
    parser.add_argument('--num_trials',
                        help='number of trials to train', type=int,
                        default=None)
    parser.add_argument('--n_lstm',
                        help='number of units in the network',
                        type=int, default=None)
    parser.add_argument('--rollout',
                        help='rollout used to train the network',
                        type=int, default=None)
    parser.add_argument('--seed', help='seed for task',
                        type=int, default=None)
    parser.add_argument('--num_cpu', help='number of threads',
                        type=int, default=None)
    parser.add_argument('--n_ch', help='number of choices',
                        type=int, default=None)

    # n-alternative task params
    parser.add_argument('--stim_scale', help='stimulus evidence',
                        type=float, default=None)

    # CV-learning task
    parser.add_argument('--th_stage',
                        help='threshold to change the stage',
                        type=float, default=None)
    parser.add_argument('--keep_days',
                        help='minimum number of sessions to spend on a stage',
                        type=int, default=None)
    parser.add_argument('--stages',
                        help='stages used for training',
                        type=int, nargs='+', default=None)

    # trial_hist wrapper parameters
    parser.add_argument('--probs', help='prob of main transition in the ' +
                        'n-alt task with trial hist.', type=float,
                        default=None)
    parser.add_argument('--block_dur',
                        help='dur. of block in the trial-hist wrappr (trials)',
                        type=int, default=None)

    # monitor wrapper parameters
    parser.add_argument('--sv_fig',
                        help='indicates whether to save step-by-step figures',
                        type=bool, default=None)
    parser.add_argument('--sv_per',
                        help='number of trials to save behavioral data',
                        type=int, default=None)
    return parser


def update_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in set(dict2).intersection(dict1))


def make_env(env_id, rank, seed=0, wrapps={}, n_args={}, **kwargs):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        for wrap in wrapps.keys():
            if not (wrap == 'Monitor-v0' and rank != 0):
                params_temp = wrapps[wrap]
                update_dict(params_temp, n_args)
                env = apply_wrapper(env, wrap, params_temp)

        return env
    set_global_seeds(seed)
    return _init


def run(alg, alg_kwargs, task, task_kwargs, wrappers_kwargs, n_args,
        rollout, num_trials, folder, n_cpu, n_lstm):
    env = test_env(task, kwargs=task_kwargs, num_steps=1000)
    num_timesteps = int(1000 * num_trials / (env.num_tr))
    if not os.path.exists(folder + 'bhvr_data_all.npz'):
        vars_ = vars()
        np.savez(folder + '/params.npz', **vars_)
        if alg == "A2C":
            from stable_baselines import A2C as algo
        elif alg == "ACER":
            from stable_baselines import ACER as algo
        elif alg == "ACKTR":
            from stable_baselines import ACKTR as algo
        elif alg == "PPO2":
            from stable_baselines import PPO2 as algo

        env = SubprocVecEnv([make_env(env_id=task, rank=i, seed=seed,
                                      wrapps=wrappers_kwargs, n_args=n_args,
                                      **task_kwargs)
                             for i in range(n_cpu)])
        model = algo(LstmPolicy, env, verbose=0, n_steps=rollout,
                     n_cpu_tf_sess=n_cpu,
                     policy_kwargs={"feature_extraction": "mlp",
                                    "n_lstm": n_lstm},
                     **alg_kwargs)
        model.learn(total_timesteps=num_timesteps)
        model.save(f"{folder}model")
        plotting.plot_rew_across_training(folder=folder)
        # TODO: save parameters in instance_folder


if __name__ == "__main__":
    # main_folder = '/gpfs/projects/hcli64/molano/neurogym/20200417/'
    main_folder = '/gpfs/projects/hcli64/shaping/'
    # main_folder = '/home/molano/priors/codes/experiments_parameters/'
    # main_folder = '/home/manuel/ngym_usage/'
    # main_folder = '/home/manuel/CV-Learning/results/'
    # get params from call
    n_arg_parser = arg_parser()
    n_args, unknown_args = n_arg_parser.parse_known_args(sys.argv)
    unkown_params = rest_arg_parser(unknown_args)
    if unkown_params:
        print('Unkown parameters: ', unkown_params)
    n_args = vars(n_args)
    n_args = {k: n_args[k] for k in n_args.keys() if n_args[k] is not None}
    main_folder = main_folder + n_args['folder'] + '/'
    name, _ = gncfd(n_args)
    instance_folder = main_folder + name + '/'
    # this is done wo the monitor wrapper's parameter folder is updated
    n_args['folder'] = instance_folder
    if not os.path.exists(instance_folder):
        os.makedirs(instance_folder)
    # load parameters
    print(main_folder+"/params.py")
    sys.path.append(os.path.expanduser(main_folder))
    spec = importlib.util.spec_from_file_location("params",
                                                  main_folder+"/params.py")
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    # update general params
    gen_params = params.general_params
    update_dict(gen_params, n_args)
    # update task params
    task_params = params.task_kwargs[gen_params['task']]
    update_dict(task_params, n_args)
    # get params
    task = gen_params['task']
    alg = gen_params['alg']
    alg_kwargs = params.algs[alg]
    seed = int(gen_params['seed'])
    num_trials = int(gen_params['num_trials'])
    rollout = int(gen_params['rollout'])
    num_cpu = int(gen_params['num_cpu'])
    n_lstm = int(gen_params['n_lstm'])
    task_kwargs = params.task_kwargs[gen_params['task']]
    run(alg=alg, alg_kwargs=alg_kwargs, task=task, task_kwargs=task_kwargs,
        wrappers_kwargs=params.wrapps, n_args=n_args, rollout=rollout,
        num_trials=num_trials, folder=instance_folder, n_cpu=num_cpu,
        n_lstm=n_lstm)
