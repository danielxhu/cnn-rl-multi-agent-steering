# E7 on Azure for Students: two extra training machines

The Windows box trains `e7_map_seed0` and `e7_map_seed1`. Two small Azure VMs
train the other two runs at the same time:

| Machine | Region | Size | Run |
|---|---|---|---|
| Windows (GTX 1070) | – | – | `e7_map_seed0`, `e7_map_seed1` |
| VM 1 | Belgium Central | Standard_D4as_v4 (4 vCPU, 16 GiB, about $0.21/h) | `e7_longgru_seed0` |
| VM 2 | Denmark East | Standard_F4s_v2 (4 vCPU, 8 GiB, about $0.19/h) | `e7_map_seed2` |

Why these: the student subscription may only use five regions, at most 6 vCPUs
per region, and most sizes are blocked (checked 16 Nov 2026). The B-series
sizes can be created but are burstable: under days of full load they fall back
to a fraction of their CPU, so they are not used.

**Estimate (not measured):** a 4-vCPU VM runs at about a third of the Windows
box, so about 20 h for `e7_map` and 15 h for `e7_longgru`; about $4–5 per
run, about $10 in total with disks and IPs.

**Keep `--workers 10`** even on 4 vCPUs. The number of workers sets the batch
of each PPO update (10 workers × 8 envs), so fewer workers would be a
different experiment from `e3_both`. Ten processes on 4 vCPUs just take turns.

Commands below for VM 1; for VM 2 change only what step 6 says.

## 1. Create the VM (portal.azure.com)

Virtual machines → Create → Azure virtual machine. Basics tab:

- Subscription: Azure for Students. Resource group: **Create new** → `e7-belgium`
  (VM 2: `e7-denmark`). Deleting this group at the end deletes everything.
- Name: `e7-belgium` (VM 2: `e7-denmark`).
- Region: **Belgium Central** (VM 2: **Denmark East**).
- Availability options: **No infrastructure redundancy required**.
- Image: **Ubuntu Server 24.04 LTS - x64 Gen2**.
- Run with Azure Spot discount: **unticked**.
- Size: **Standard_D4as_v4** (VM 2: **Standard_F4s_v2**).
- Authentication type: **Password**. Username `azureuser`, choose a password.
- Public inbound ports: **Allow selected ports**, **SSH (22)**.

Disks tab: OS disk type **Standard SSD**. Everything else default. Then
Review + create → Create. When it finishes, open the VM and copy its
**Public IP address**.

## 2. Connect (Windows command prompt)

```
ssh azureuser@<public-ip>
```

Answer `yes` the first time, then type the password. Everything in steps 3–6
runs inside this connection.

## 3. Set up (once, about 5 min)

```
git clone https://github.com/danielxhu/cnn-rl-multi-agent-steering.git
bash cnn-rl-multi-agent-steering/policy/scripts/azure_setup.sh
```

It must end with `11/11 passed` and `setup done: 4 vCPUs`.

## 4. Upload the validation suite (second Windows command prompt)

```
scp -r E:\policy-data\val azureuser@<public-ip>:~/policy-data/
```

Back in the SSH window, check: `ls ~/policy-data/val | head` lists `suite.json`.

## 5. Start a tmux session

Training must survive closing the SSH window, so it runs inside tmux:

```
tmux new -s e7
```

## 6. Train

VM 1 (Belgium, `e7_longgru_seed0`):

```
cd ~/cnn-rl-multi-agent-steering/policy
~/venv/bin/python train.py --memory gru --seed 0 --out ~/policy-runs/e7_longgru_seed0 --val-suite ~/policy-data/val --val-workers 2 --workers 10 --device cpu --timeout-penalty 25 --progress-metric geodesic --L 128
```

VM 2 (Denmark, `e7_map_seed2`):

```
cd ~/cnn-rl-multi-agent-steering/policy
~/venv/bin/python train.py --memory gru --seed 2 --out ~/policy-runs/e7_map_seed2 --val-suite ~/policy-data/val --val-workers 2 --workers 10 --device cpu --timeout-penalty 25 --progress-metric geodesic --wallmap on
```

Leave tmux running with **Ctrl+B, then D**. You can now close the SSH window.
To look again later: `ssh azureuser@<public-ip>`, then `tmux attach -t e7`.

The `st/s` value of the first lines gives the real speed: hours ≈ 50,000,000
÷ st/s ÷ 3600.

**If the VM restarted or training stopped:** start tmux again and run the same
command with `--resume` at the end. It continues from the last checkpoint
(saved every 25 iterations).

## 7. When the run is finished

The run is finished when the log says it reached 50,000,000 agent-steps and
the prompt is back. From a Windows command prompt:

```
scp -r azureuser@<public-ip>:~/policy-runs/e7_longgru_seed0 E:\policy-runs\
```

(VM 2: `e7_map_seed2`.) Check that `E:\policy-runs\e7_longgru_seed0\ckpt_best.pt`
exists.

## 8. Delete the VM (stops all charges)

Portal → Resource groups → `e7-belgium` (VM 2: `e7-denmark`) → **Delete
resource group**, and type its name to confirm. A stopped VM still costs money
for its disk and IP; deleting the group removes all of it. Only do this after
step 7 has copied the run.

## 9. Windows: your share, then everything else

While the VMs train:

```
cd C:\Users\danie\cnn-rl-multi-agent-steering
git pull
cd policy
python scripts/run_e7.py --train-only e7_map_seed0 e7_map_seed1
```

When all four runs are in `E:\policy-runs` (yours plus the two copied back):

```
python scripts/run_e7.py --full
```

It skips the pilot and the finished runs (each exits at once with
`--resume`), then evaluates, aggregates and pushes `policy/results/e7`.
