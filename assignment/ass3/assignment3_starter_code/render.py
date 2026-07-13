import argparse
import pathlib
from typing import Union
from collections import defaultdict
from PIL import Image, ImageDraw

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from moviepy.editor import ImageSequenceClip
from run_dpo import SFT, DPO


def hopper_state_from_observation(env: gym.Env, obs: np.ndarray):
    qpos = env.unwrapped.data.qpos.flat.copy()
    qpos[1:] = obs[:5]
    qvel = obs[5:]

    return qpos, qvel


def _render_episode(env, seed, policy):
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    np.random.seed(seed + 3)

    images = []
    images.append(env.render())
    rewards = []
    for _ in range(env.spec.max_episode_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        images.append(env.render())

        if done:
            break

    return images, rewards


def render_dpo():
    assert args.checkpoint.endswith("dpo.pt")
    dpo_path = args.checkpoint
    sft_path = dpo_path.replace("dpo.pt", "sft.pt").replace("Hopper-v4-dpo", "Hopper-v4-sft")
    print("loading dpo from:", dpo_path)
    print("loading sft from:", sft_path)

    dpo: DPO = torch.load(dpo_path, weights_only=False)
    sft: SFT = torch.load(sft_path, weights_only=False)

    sft_returns = []
    dpo_returns = []
    for ep in range(args.num_episode):
        sft_images, sft_rewards = _render_episode(env, ep + 1, sft.act)
        dpo_images, dpo_rewards = _render_episode(env, ep + 1, dpo.act)
        print(
            f"episode {ep}: sft reward: {np.sum(sft_rewards):.1f}, "
            f"dpo reward: {np.sum(dpo_rewards):.1f}"
        )

        combined_images = []
        for i in range(len(sft_images)):
            combined = np.hstack((sft_images[i], dpo_images[i]))
            combined_images.append(combined)

        clip = ImageSequenceClip(combined_images, fps=25)
        clip.write_videofile(
            pathlib.Path(args.checkpoint).parent.joinpath(f"eval_ep{ep}.mp4").as_posix(),
            verbose=False,
            logger=None,
        )

        sft_returns.append(np.sum(sft_rewards))
        dpo_returns.append(np.sum(dpo_rewards))
        print("Video saved to:", pathlib.Path(args.checkpoint).parent.joinpath(f"eval_ep{ep}.mp4"))

    print(f"Average rewards: sft: {np.mean(sft_returns):.1f}, dpo: {np.mean(dpo_returns):.1f}")


def render_ppo():
    assert args.checkpoint.endswith(".zip")
    agent = sb3.PPO.load(args.checkpoint)

    def policy(obs):
        return agent.predict(obs)[0]

    ppo_returns = []
    for ep in range(args.num_episode):
        ppo_images, ppo_rewards = _render_episode(env, ep + 1, policy)
        print(f"episode {ep}: RLHF reward: {np.sum(ppo_rewards):.1f}")

        clip = ImageSequenceClip(ppo_images, fps=25)
        clip.write_videofile(
            pathlib.Path(args.checkpoint).parent.joinpath(f"eval_ep{ep}.mp4").as_posix(),
            verbose=False,
            logger=None,
        )

        ppo_returns.append(np.sum(ppo_rewards))

    print(f"PPO: {np.mean(ppo_returns):.1f}")


def render_dataset():
    video_folder = pathlib.Path(args.dataset).parent.joinpath("videos")
    if not video_folder.exists():
        video_folder.mkdir()

    data = np.load(args.dataset, allow_pickle=True)
    if args.idx is None:
        idx = np.random.randint(data["obs_1"].shape[0])
    else:
        idx = int(args.idx)
    print(f"rendering {idx}/{data['obs_1'].shape[0]}")

    label = data["label"][idx].item()
    assert label in (0, 0.5, 1)
    if label == 0.5:
        labels = ["Equally Preferred", "Equally Preferred"]
    elif label == 1:
        labels = ["Not Preferred", "Preferred"]
    else:
        labels = ["Preferred", "Not Preferred"]

    frames = defaultdict(list)
    for obs_key in ["obs_1", "obs_2"]:
        env.reset()
        for obs in data[obs_key][idx]:
            env.unwrapped.set_state(*hopper_state_from_observation(env, obs))
            frames[obs_key].append(env.render())
    combined_frames = []
    for frame1, frame2 in zip(frames["obs_1"], frames["obs_2"]):
        combined_img = Image.fromarray(np.hstack((frame1, frame2)))
        width = combined_img.size[0]
        draw = ImageDraw.Draw(combined_img)
        draw.text((int(width * 0.25), 0), labels[0])
        draw.text((int(width * 0.75), 0), labels[1])
        combined_frames.append(np.array(combined_img))
    clip = ImageSequenceClip(combined_frames, fps=int(1 / (5 * env.dt)))
    clip.write_videofile(video_folder.joinpath(f"pair_{idx}.mp4").as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="The model checkpoint")
    parser.add_argument("--num_episode", type=int, default=10, help="The model checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="The preference dataset")
    parser.add_argument("--idx", type=int, default=None, help="the index of the demo in dataset")
    parser.add_argument("--dpo", action="store_true")
    args = parser.parse_args()

    env = gym.make("Hopper-v4", terminate_when_unhealthy=False, render_mode="rgb_array")

    if args.dpo:
        render_dpo()
    elif args.dataset:
        render_dataset()
    else:
        render_ppo()
