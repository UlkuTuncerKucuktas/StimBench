# StimBench-Syn: handoff on the arm-flapping problem

## Goal

StimBench is a benchmark of YouTube clips of autistic children showing three
stimming behaviours (ArmFlapping, HeadBanging, Spinning) plus a Normal class.
The journal extension adds a synthetic set, StimBench-Syn, generated with
Wan 2.2 T2V-A14B so that the paper can test synthetic-only, real-only and mixed
training. The set must be controlled and documented (gender, age, room,
camera, severity, movement variant, prompt and seed recorded per clip) and must
not condition on any real child footage.

HeadBanging, Spinning and Normal clips are accepted by the domain expert.
ArmFlapping is not solved: the generated clips mostly look like a child
standing or bouncing, not flapping.

## What real arm flapping looks like (expert's words)

Arms hang low and swing from the elbows; the hands are limp and "ragdollish",
whirling loosely round the wrists, not stiff and not fists; the child is
usually excited by something (screen, toy, song); feet stay planted, no
jumping. One generated clip got it right: `smoke_v11/ArmFlapping/wan2.2-14b-480p_ArmFlapping_0002.mp4`
(md5 0cd42f6a...). By a MediaPipe hand/pose metric it has about twice the
wrist-flexion amplitude (111 deg) and elbow swing (150 deg) of clips made from
the same wording, so it was a lucky draw of the noise, not a wording effect.

## Generation stack (all in this repository)

- `gen_synth.py` — CLI: `plan | generate | report | motion | hands | frames | i2v | v2v | vace | flow`.
- `stimbench/synth/vocab.py` — every prompt pool: movement variants per class,
  severity texts keyed by (class, posture), environments, cameras, child
  appearance, per-class negative prompts, class weights
  (`SEVERITY_WEIGHTS_BY_CLASS`, `TOPOGRAPHY_WEIGHTS`), `LIMP` hand description.
- `stimbench/synth/sampler.py` — factorial sampler (movement x severity x
  tempo, dealt decks for scene/camera/child, gender parity), prompt renderer,
  A/B plan mode (`ab:` config, fixed scene, only the movement sentence varies).
- `stimbench/synth/audit.py` — plan checks (no negations, tempo clause,
  balance, duplicates, token limit <= 500 of the 512 T5 budget).
- `stimbench/synth/generate.py` — resumable Wan 2.2 T2V generation
  (diffusers `WanPipeline`, two experts, guidance 4.0/3.0, UniPC flow_shift 3,
  480x832, 81 frames at 16 fps, 40 steps, torch.compile), per-clip seed and
  plan hash, manifest.jsonl.
- `stimbench/synth/video.py` — ffmpeg retime: clips are generated at half
  tempo and played back at 32 fps so fast movements survive the 16 fps prior.
- `stimbench/synth/motion.py` — whole-frame motion energy, freeze fraction,
  periodicity from the vertical centroid autocorrelation (`motion.csv`).
- `stimbench/synth/hands.py` — MediaPipe finger curl, wrist flexion amplitude,
  elbow swing, wrist lag (`hands.csv`; needs mediapipe 0.10.14, runs on a Mac).
- `stimbench/synth/i2v.py` — Wan 2.2 I2V from a first frame (parked: the
  pod lacks `ftfy`, which diffusers' I2V pipeline imports unguarded).
- `stimbench/synth/v2v.py` — SDEdit on the T2V experts: re-noise a source
  clip's VAE latents to a strength and denoise under a new prompt.
- `stimbench/synth/vace.py` — Wan2.1-VACE-14B driven by an optical-flow control
  video of the source clip; `flow` renders the control videos with RAFT
  (torchvision) on another host.
- `configs/synth/wan22_a14b_480p_fast.yaml` — release config (520 clips, 130 per class).
  `ab_*.yaml`, `v2v_armflapping.yaml`, `vace_armflapping.yaml`, `i2v_armflapping.yaml` — the experiments below.
- `run_synth.sh`, `watch_synth.sh` — detached launcher and watchdog.

Compute: one H200 on a Run:AI pod; 242 s per clip with compile. The pod has
torch 2.13+cu130, diffusers 0.39, imageio-ffmpeg; no torchvision, cv2,
mediapipe or ftfy, and installs there are avoided because a wrong wheel could
break the working generator.

## What was tried for ArmFlapping, and what happened

1. Wording, five revisions (smoke_v11 to v15, 8 clips each): caption-register
   sentences, puppet/limp-hand simile, "flick loosely at the wrists", explicit
   excitement triggers, negatives for fists and stiff wrists, then "feet
   planted, arms swinging big, hands whirling round the wrists" with jumping,
   hopping and bouncing in the negative prompt. Resolved-periodicity rate never
   moved (0-3 of 8); the expert judged most clips as Normal-looking or bouncing.
2. Severity and variant weighting: ArmFlapping set to 90 % the base variant
   (`af_sides`) and 90 % pronounced/whole-body. Distribution is exactly as
   configured, but the more overt tiers do not produce more arm motion; whole-
   body tiers produced jumping, which the expert rejected.
3. A/B sweeps with a fixed scene (`ab_armflapping.yaml`, 10 wording arms;
   `ab_armflapping_amplitude.yaml`, 8 amplitude/count arms in the chosen clip's
   own scene): no arm separates from the others by motion or hand metrics.
   Note: the amplitude sweep varied the seed with the arm; fixed since
   (A/B arms now share the base clip's seed via `seed_override`).
4. SDEdit video-to-video from the chosen clip (`v2v_armflapping.yaml`, strengths
   0.5/0.65/0.8, three new-child prompts): movement kept but the output is
   "always the same video"; appearance rides with the motion in the latents.
5. Flow-control with VACE (`vace_armflapping.yaml`): RAFT flow of the chosen
   clip (plain and mirrored) as the control video, new child/scene from the
   prompt, arms at conditioning scale 1.0 / 0.7 / mirrored. Model downloaded
   and loads on the pod; control videos rendered; the run has not been done.
6. Parked: Wan 2.2 I2V from a generated first frame (needs `ftfy`);
   Wan2.2-Animate with a driving video (needs pose preprocessing the pod cannot
   run); motion LoRA trained on real clips (rejected: conditions a released set
   on real children).

## Measurements that matter

- Chosen clip vs same-wording clips (MediaPipe): wrist flexion 111 vs 48-92
  deg, elbow swing 150 vs 57-94 deg, fingers open (curl 164). Real flapping
  clips: finger curl 79-150, wrist flexion 31-145.
- Whole-frame motion energy does not separate flapping from bouncing, and the
  periodicity metric in `motion.csv` cannot see hands: it scored the expert's
  chosen clip among the lowest of its set (peak 0.20). Do not tune flapping
  against the resolved rate; use the MediaPipe wrist and elbow amplitudes,
  which run off the pod.
- Seed variance dominates wording: identical wording at two seeds gave 1/8 and
  3/8 resolved periods. Across every flapping attempt (v11 1/8, v12 1/8, v13
  3/8, v14 0/8, v15 1/8, ab_af 1/10, ab_af2 0/8: 59 clips, 7 resolved) no
  wording change moved the number in either direction. Single-clip comparisons
  cannot rank wordings; a fresh sweep must hold the seed fixed across arms
  (`seed_override` in the A/B plan does this now, `ab_af` and the first
  `ab_af2` run did not).

## Open question for the next attempt

How to make Wan produce loose, large-amplitude flapping with whirling hands
reliably enough for 130 diverse clips (different children, rooms, cameras),
without conditioning on real footage. Candidates not yet exhausted: VACE flow
or pose control from a few approved synthetic clips; a motion LoRA trained on
approved synthetic clips only; rejection sampling with the hand metric (about
one in ten to twenty draws is acceptable); a different generator.
