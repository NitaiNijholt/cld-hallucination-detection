# AGENTS.md

Developer notes for AI agents and human contributors working in this repository.

---

## Connecting to Snellius (SURF HPC)

Snellius is the national Dutch supercomputer used for GPU finetuning jobs.
Username: `nnijholt`

### Prerequisites

1. **Accept the SURF Usage Agreement** at https://portal.cua.surf.nl (one-time, required before anything works)
2. **SSH key registered** at https://portal.cua.surf.nl/user/keys
   - Local key lives at `~/.ssh/id_ed25519`
   - Fingerprint: `SHA256:PA0kF0xe9pBkcz4uXa3l6or2C6L+sbc7PkegIaoo7jI`

### SSH login

Direct SSH to `snellius.surf.nl` requires a whitelisted IP. From home/WSL2, use the **doornode** instead:

```bash
ssh nnijholt@doornode.hpcv.surf.nl
# select "Snellius" from menu, enter Snellius password
```

Once your home IP is whitelisted (email helpdesk@surfsara.nl with your IP from https://echoip.cua.surf.nl), direct access works:

```bash
ssh snellius   # uses ~/.ssh/config alias
```

`~/.ssh/config` entry (already configured locally):

```
Host snellius
    HostName snellius.surf.nl
    User nnijholt
    ProxyJump nnijholt@doornode.hpcv.surf.nl
    IdentityFile ~/.ssh/id_ed25519

Host doornode
    HostName doornode.hpcv.surf.nl
    User nnijholt
    IdentityFile ~/.ssh/id_ed25519
```

### File transfer

The doornode blocks SCP port-forwarding. Generate data files **on Snellius** directly:

```bash
# On Snellius — data is already in the cloned repo (final_runs/)
cd ~/cld-hallucination-detection
source venv/bin/activate
python -m finetuning.src.data.prepare_judge_data
```

Once your IP is whitelisted, direct SCP works:

```bash
scp localfile.txt nnijholt@snellius.surf.nl:~/destination/
```

### Environment setup (one-time on Snellius)

```bash
git clone -b snellius-finetuning https://github.com/NitaiNijholt/cld-hallucination-detection ~/cld-hallucination-detection
cd ~/cld-hallucination-detection

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv venv --python 3.11
source venv/bin/activate
pip install -e finetuning/
```

### Submitting and monitoring jobs

```bash
cd ~/cld-hallucination-detection
source venv/bin/activate

# Prepare data (only needed once)
python -m finetuning.src.data.prepare_judge_data

# Submit training job
mkdir -p finetuning/logs
sbatch finetuning/jobs/train_judge.job

# Monitor
squeue -u nnijholt
tail -n 80 finetuning/logs/cld_judge_finetune_<JOBID>.out

# Check SBU usage
accuse
```

### SBU cost estimate

| Resource | Rate | 3h job cost |
|---|---|---|
| 1× H100 GPU | ~192 SBU/hr | ~576 SBU |
| Budget | 150,000 SBU | ~0.4% per run |

### Troubleshooting

| Problem | Fix |
|---|---|
| `No route to host` | IP not whitelisted — use doornode |
| `You have not accepted the Usage Agreement` | Accept at https://portal.cua.surf.nl |
| `tail -f` terminal frozen | Open new terminal, reconnect via doornode |
| `venv/bin/activate: No such file or directory` | Run `uv venv venv --python 3.11 --clear` first |
| `channel 0: open failed: administratively prohibited` | Doornode blocks SCP — generate files on Snellius directly |
